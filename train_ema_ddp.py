import argparse 
import os 
import torch 
from torch import optim 

from transformers import AutoTokenizer
from open_clip import CLIPVisionCfg, CLIPTextCfg, ClipLoss 
from open_clip import Union, create_transform 
from open_clip import EMA 
from utils import get_cc3m_dataset, reconstruct_loss, get_imagenet_dataset  
from torch.cuda.amp import GradScaler

from training.distributed import is_master, init_distributed_device, world_info_from_env 
from training.precision import get_autocast
from training.train import evaluate
from training.scheduler import cosine_lr 
from utils import zero_shot_classifier, zero_shot_run
from tqdm import tqdm 
from training import imagenet_classnames, openai_imagenet_template 

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import torch.utils.tensorboard as tensorboard

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/mayuchen/datasets/research/conceptual_captions',
        help="Path to file(s) with training data",
    )
    parser.add_argument(
        "--imagenet",
        type=str,
        default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/fanmingyuan/cephfs_bk/data/ILSVRC2012',
        help="Path to imagenet data for zero-shot accuracy",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.98, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--norm_gradient_clip", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bfloat16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument("--checkpoint-path", type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/feizhengcong/clip/union', help='checkpoint path')
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank.")
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--tensorboard_path", type=str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/feizhengcong/clip/union/tensorboard/ema",
    )
    parser.add_argument("--debug", type=bool, default=False) 

    args = parser.parse_args()
    return args 



def main(): 
    args = parse_args() 
    print(args) 

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    if is_master(args):
        if not os.path.exists(args.tensorboard_path): 
            os.makedirs(args.tensorboard_path)
        writer = tensorboard.SummaryWriter(args.tensorboard_path)
    else:
        writer = None
    
    preprocess_train, preprocess_val = create_transform(image_size=224) 

    tokenizer = AutoTokenizer.from_pretrained('ckpt/tokenizer')
    model = Union(embed_dim=768, vision_cfg=CLIPVisionCfg(), text_cfg=CLIPTextCfg()) 
    model = model.to(device) 

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device],) #find_unused_parameters=True)

    ema_model = EMA(model, )

    # create optimizer and scaler
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p) 

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    scaler = GradScaler() if args.precision == "amp" else None

    train_loader = get_cc3m_dataset(args, preprocess_train, is_train=True, tokenizer=tokenizer) 
    val_loader = get_cc3m_dataset(args, preprocess_val, is_train=False, tokenizer=tokenizer)
    imagenet_loader = get_imagenet_dataset(args, preprocess_val, is_train=False)

    clip_loss = ClipLoss(local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod)


    for epoch in range(32): 
        model.train() 
        device = torch.device(args.device)
        autocast = get_autocast(args.precision) 

        num_batches_per_epoch = train_loader.num_batches
        loss_cum = .0 
        progress = tqdm(total=len(train_loader), desc='Union training') 
        for i, batch in enumerate(train_loader):  
            step = num_batches_per_epoch * epoch + i

            images, texts = batch 
            images = images.to(device=device)
            texts = texts.to(device=device)
            optimizer.zero_grad()

            with autocast():
                image_features, text_features, logit_scale = model(images, texts, combine=False) 
                image_text_features, _ = ema_model(images, texts, combine=True) 
                image_text_features = image_text_features.detach() 

                loss_1 = clip_loss(image_features, text_features, logit_scale) 
                loss_2 = clip_loss(image_features, image_text_features, logit_scale) 
                loss_3 = clip_loss(text_features, image_text_features, logit_scale)
                total_loss = loss_1 + loss_2 + loss_3
            if scaler is not None:
                scaler.scale(total_loss).backward()
     
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
                scaler.update()
            else: 
                total_loss.backward() 
                optimizer.step() 
            loss_cum += total_loss.item() 
            progress.set_postfix({"loss": loss_cum / (i + 1)})
            progress.update() 
            if is_master(args) and  i % 10 == 0: 
                writer.add_scalar("train/loss", total_loss.item(), step)
                writer.add_scalar("train/loss_it", loss_1.item(), step)
                writer.add_scalar("train/loss_ic", loss_2.item(), step)
                writer.add_scalar("train/loss_tc", loss_3.item(), step)
                # writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step) 
            
            if args.debug == True:
                break 

        model.eval() 
        with torch.no_grad(): 
            progress = tqdm(total=len(val_loader), desc='Union evaluation') 
            acc_cum = .0
            for i, batch in enumerate(val_loader):
                images, texts = batch 
                images = images.to(device=device)
                texts = texts.to(device=device) 
                
                image_features, text_features, logit_scale = model(images, texts) 
                logits = torch.matmul(text_features, image_features.t()) * logit_scale 
                pred = torch.argmax(logits, dim=-1) 
                accuracy = torch.eq(pred, torch.arange(len(logits), device=logits.device)).sum() / len(logits)
                acc_cum += accuracy.item() 
                progress.set_postfix({"accuracy": acc_cum / (i + 1)})
                progress.update()

                if args.debug == True:
                    break 
            if is_master(args):
                writer.add_scalar("val/retrieval_acc", acc_cum / len(val_loader), epoch) 

        # zero-shot for imagenet accuracy
        data = {} 
        data["imagenet-val"] = imagenet_loader
        print('building zero-shot classifier')
        evaluate(model, data, epoch+1, args, writer, tokenizer) 
        '''
        classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, tokenizer, device)
        acc1, acc5 = zero_shot_run(model, classifier, imagenet_loader, device)
        print('acc1: ', acc1) 

        if is_master(args):
            writer.add_scalar("val/imagenet_acc_1", acc1, epoch) 
            writer.add_scalar("val/imagenet_acc_5", acc5, epoch) 
        '''
        if is_master(args):
            print('save modeling')
            torch.save(model.state_dict(), '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/feizhengcong/clip/union/ckpt/' + str(epoch) + '.pt') 
            torch.cuda.synchronize()



if __name__ == "__main__":
    main()
