#!/bin/bash
# JiT-B/16 on native field of view panoramas (16 px patches), warm-started from the JiT-B/32 run.
# - --fused_attn: memory-efficient attention, required (the explicit attention matrix does not fit at 2816 px width)
# - --turn_tokens: defaults to 96 for 16 px patches (48 for 32 px)
# - --init_from: copies every weight whose shape matches (transformer blocks, weather and time embedders);
#   the patch embedding and the output layer start fresh
# - batch 32, blr 1.5e-4 -> lr 1.875e-5 (same as epochs 100-199 of the B/32 run)
# Add --grad_ckpt if the widest batches run out of memory (about 30% slower).

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 --fused_attn \
--init_from ./results/jit_native --init_key model_ema1 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--online_eval --skip_fid --eval_freq 1 \
--num_images 16 --gen_bsz 16 --grid_size 4 --sample_ema 2 \
--noise_scale 1.0 \
--batch_size 32 --blr 1.5e-4 \
--epochs 100 --warmup_epochs 1 --steps_per_epoch 5000 --save_last_freq 1 \
--num_workers 18 --prefetch_factor 2 \
--weather_patch_size 4 --cond_drop_prob 0.05 --cond_drop_both_prob 0.05 --weather_drop_prob 0.1 \
--output_dir ./results/jit_native_p16 --resume ./results/jit_native_p16 \
--data_path ./data
