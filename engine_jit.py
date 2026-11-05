import math
import sys
import os
import shutil
import json

import torch
import torchvision
from torch.utils.data import default_collate
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # (I0, It, It_prev, w0, wt, wt_prev, angle_mask, view_mask)
        tensors = [t.to(device, non_blocking=True) for t in batch["tensors"]]
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(*tensors)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def load_preview_sample(dataset, index, seed):
    """Loads a dataset sample with a fixed random crop, without touching the global RNG"""
    with torch.random.fork_rng():
        torch.manual_seed(seed + index)
        return dataset[index]


def evaluate(model_without_ddp, dataset, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = math.ceil(args.num_images / (batch_size * world_size))

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-image{}-res{}-ema{}-epoch{:03d}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale, args.num_images, args.img_size,
            args.sample_ema, epoch
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to the ema params selected by --sample_ema ("none" keeps the raw weights)
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    if args.sample_ema != "none":
        ema_params = model_without_ddp.ema_params1 if args.sample_ema == "1" else model_without_ddp.ema_params2
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema{}".format(args.sample_ema))
        model_without_ddp.load_state_dict(ema_state_dict)

    # fixed preview samples (same seed every epoch) so forecasts are comparable across epochs
    preview_idx = np.random.default_rng(args.seed).choice(len(dataset), min(args.num_images, len(dataset)), replace=False)
    preview_idx = np.hstack([preview_idx, np.zeros(world_size * batch_size * num_steps, dtype=int)])
    if misc.is_main_process():
        with open(save_folder + "-preview.json", "w") as f:
            json.dump([{"view": dataset[int(k)]["view"], "datetimes": dataset[int(k)]["datetimes"]}
                       for k in preview_idx[:min(args.grid_size, args.num_images)]], f, indent=2)

    grid_images = []
    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        batch = default_collate([load_preview_sample(dataset, int(k), args.seed) for k in preview_idx[start_idx:end_idx]])
        I0, It, It_prev, w0, wt, wt_prev, angle_mask, view_mask = [t.cuda() for t in batch["tensors"]]

        # fixed noise per batch so samples are comparable across epochs; fork_rng leaves the training RNG untouched
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(args.seed + world_size * i + local_rank)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                sampled_images = model_without_ddp.generate(I0, It_prev, w0, wt, wt_prev, angle_mask, view_mask)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            if len(grid_images) < 4 * args.grid_size:
                # one grid row per sample: It_prev | I0 | It (ground truth) | It (forecast)
                for frame in (It_prev[b_id], I0[b_id], It[b_id]):
                    grid_images.append(((frame.float().cpu() + 1) / 2).clamp(0, 1))
                grid_images.append(sampled_images[b_id].clamp(0, 1))
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # sample grid from rank 0, saved next to the image folder so it survives the FID cleanup
    if misc.is_main_process() and len(grid_images) > 0:
        grid = torchvision.utils.make_grid(grid_images, nrow=4, padding=2)
        torchvision.utils.save_image(grid, save_folder + "-grid.png")
        if log_writer is not None:
            log_writer.add_image('samples', grid, epoch)

    # compute FID and IS
    if log_writer is not None and not args.skip_fid:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        shutil.rmtree(save_folder)

    torch.distributed.barrier()
