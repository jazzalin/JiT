#!/bin/bash
# JiT-B/32 on native field of view panoramas, conditioned on a clear-sky reference frame (data.reference in
# config.yml, from the clear_score_index.py outputs), warm-started from the JiT-B/32 run without reference.
# - --init_from: copies every weight; the input patch embedding gets 4 new channels (Iref RGB + reference age),
#   initialised to zero, so training starts from the forecasts of the source run
# - --ref_drop_prob: the reference is replaced by the missing reference (black, age -1) in 20% of the samples
#   (on top of the samples without a clear frame in the last 14 days), so the model also works without it
# For the 16 px patch model, use --model JiT-B/16 --fused_attn and --init_from ./results/jit_native_p16.

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--init_from ./results/jit_native --init_key model_ema1 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--online_eval --skip_fid --eval_freq 1 \
--num_images 16 --gen_bsz 16 --grid_size 4 --sample_ema 2 \
--noise_scale 1.0 \
--batch_size 64 --blr 5e-5 \
--epochs 100 --warmup_epochs 1 --steps_per_epoch 5000 --save_last_freq 1 \
--weather_patch_size 4 --cond_drop_prob 0.05 --cond_drop_both_prob 0.05 --weather_drop_prob 0.1 \
--ref_drop_prob 0.2 \
--output_dir ./results/jit_native_ref --resume ./results/jit_native_ref \
--data_path ./data
