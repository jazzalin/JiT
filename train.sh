#!/bin/bash

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--online_eval --skip_fid --eval_freq 1 \
--num_images 16 --gen_bsz 16 --grid_size 4 --sample_ema 2 \
--noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 100 --warmup_epochs 1 --steps_per_epoch 5000 --save_last_freq 1 \
--weather_patch_size 4 --cond_drop_prob 0.05 --cond_drop_both_prob 0.05 --weather_drop_prob 0.1 \
--output_dir ./results/jit_pano --resume ./results/jit_pano \
--data_path ./data --online_eval