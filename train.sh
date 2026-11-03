#!/bin/bash

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--online_eval --skip_fid --eval_freq 1 \
--num_images 16 --gen_bsz 16 --grid_size 16 --preview_classes 4 --sample_ema 2 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 100 --warmup_epochs 1 --steps_per_epoch 5000 --save_last_freq 1 \
--cfg 2.0 --interval_min 0.0 --interval_max 1.0 \
--output_dir ./results/jit_cam --resume ./results/jit_cam \
--data_path ./data --online_eval