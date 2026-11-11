import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from util.crop import center_crop_arr
import util.misc as misc
from util.sampler import RandomSubsetSampler, WidthBinBatchSampler

import copy
from engine_jit import train_one_epoch, evaluate

from denoiser import Denoiser
import yaml
from photocast_dataloader import get_photocast_dataloader


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=None, type=int, nargs='+',
                        help='Image size: S for SxS crops, or H W (e.g. 256 2048 for full panoramas). '
                             'Default: taken from the data (data.img_size / data.train_crop in the config); '
                             'if given, it must match the data')
    parser.add_argument('--circular_rope', action='store_true',
                        help='Column RoPE and sincos pos_embed periodic over the width (360 deg panoramas wrap around)')
    # --cylindrical is set automatically for native field of view data (data.train_crop: native)
    parser.add_argument('--cylindrical', action='store_true',
                        help='Panoramas of any width: position embedding and column RoPE from each column azimuth')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--steps_per_epoch', default=0, type=int,
                        help='If > 0, each epoch is this many steps on a fresh random subset of the data (0 = full pass)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--cond_drop_prob', default=0.05, type=float,
                        help='Probability of blacking out I0 and It_prev, independently, during training')
    parser.add_argument('--cond_drop_both_prob', default=0.05, type=float,
                        help='Probability of blacking out I0 and It_prev together during training')
    parser.add_argument('--weather_drop_prob', default=0.1, type=float,
                        help='Probability of replacing the weather by the learned null weather during training (weather CFG)')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=40, type=int)
    parser.add_argument('--prefetch_factor', default=2, type=int,
                        help='Batches prefetched per worker: num_workers * prefetch_factor batches sit in host memory')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Weather classifier-free guidance scale for sampling (1.0 = no guidance)')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')
    parser.add_argument('--skip_fid', action='store_true',
                        help='Only save generated images and a sample grid, skip FID/IS computation')
    parser.add_argument('--grid_size', type=int, default=16,
                        help='Number of forecasts in the logged sample grid (one row each: It_prev, I0, It, forecast)')
    parser.add_argument('--sample_ema', type=str, default='1', choices=['1', '2', 'none'],
                        help='Weights used for generation: ema_params1, ema_params2 or the raw (non-EMA) weights')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--weather_patch_size', default=2, type=int,
                        help='Patch size (in weather grid cells) for tokenizing the weather fields')
    parser.add_argument('--config', type=str, default="config.yml")

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    with open(args.config, "r", encoding="utf-8") as fid:
        config = yaml.safe_load(fid)

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # TensorBoard logging disabled
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    log_writer = None

    dataset_train = get_photocast_dataloader(config)

    # conditioning sizes from one sample: (I0, It, It_prev, w0, wt, wt_prev, angle_mask, view_mask)
    I0, _, It_prev, w0, wt, wt_prev, angle_mask, view_mask = dataset_train[0]["tensors"]
    data_size = list(I0.shape[1:])
    native = getattr(dataset_train, "widths", None) is not None
    if native:
        # native field of view: one width per camera, the model takes any width (multiple of the patch size)
        args.cylindrical = True
        assert not args.circular_rope, "--circular_rope does not apply to native field of view panoramas"
        widths = np.unique(dataset_train.widths)
        print("Native field of view panoramas: height {}, {} widths from {} to {} (cylindrical model)".format(
            data_size[0], len(widths), widths.min(), widths.max()))
        img_size = [data_size[0], data_size[0]]  # the model's fixed size is the height only
    elif args.img_size is None:
        img_size = data_size
    else:
        img_size = args.img_size * 2 if len(args.img_size) == 1 else args.img_size
        assert data_size == img_size, \
            "--img_size {} does not match the data ({}): drop --img_size or set data.img_size / data.train_crop in {}".format(
                img_size, data_size, args.config)
    print("Image size: {}x{}".format(*img_size))
    args.img_size = img_size[0] if img_size[0] == img_size[1] else img_size
    args.cond_channels = I0.shape[0] + It_prev.shape[0] + angle_mask.shape[0] + view_mask.shape[0]
    args.weather_channels = w0.shape[0]  # per time step, the embedder is shared across (wt, w0, wt_prev)
    args.weather_times = 3
    assert w0.shape[1] == w0.shape[2], "weather crops must be square, got {}".format(tuple(w0.shape[1:]))
    args.weather_size = w0.shape[1]
    print("Conditioning: {} frame channels, {} x {} weather channels on a {}x{} grid ({} tokens)".format(
        args.cond_channels, args.weather_times, args.weather_channels, args.weather_size, args.weather_size,
        args.weather_times * (args.weather_size // args.weather_patch_size) ** 2))

    # record of the cameras that loaded
    if global_rank == 0 and args.output_dir:
        with open(os.path.join(args.output_dir, "cameras.json"), "w") as f:
            json.dump(dataset_train.cameras, f, indent=2)

    batch_sampler_train = None
    if native:
        # batches of one width; fixed-length epochs with steps_per_epoch > 0
        batch_sampler_train = WidthBinBatchSampler(
            dataset_train.widths, args.batch_size, num_batches=args.steps_per_epoch or None,
            num_replicas=num_tasks, rank=global_rank, seed=args.seed
        )
        sampler_train = None
    elif args.steps_per_epoch > 0:
        # fixed-length epochs: a new random subset of steps_per_epoch batches per GPU each epoch
        sampler_train = RandomSubsetSampler(
            dataset_train, num_samples=args.steps_per_epoch * args.batch_size,
            num_replicas=num_tasks, rank=global_rank, seed=args.seed
        )
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    print("Sampler_train =", batch_sampler_train or sampler_train)

    loader_kwargs = dict(num_workers=args.num_workers, persistent_workers=True,
                         prefetch_factor=args.prefetch_factor, pin_memory=args.pin_mem)
    if batch_sampler_train is not None:
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_train, **loader_kwargs)
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train, batch_size=args.batch_size, drop_last=True, **loader_kwargs)

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                evaluate(model_without_ddp, dataset_train, args, 0, batch_size=args.gen_bsz, log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # both samplers reshuffle per epoch only through set_epoch, so call it even on a single GPU
        (batch_sampler_train or data_loader_train.sampler).set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 10 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, dataset_train, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
