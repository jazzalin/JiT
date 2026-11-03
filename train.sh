#!/bin/bash

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--class_num 1 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 10 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 1.0 --interval_min 1.0 --interval_max 1.0 \
--output_dir ./results --resume ./results \
--data_path ./data --online_eval