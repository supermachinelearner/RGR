#!/bin/bash

BASE_MODEL=""
EXP_DIR=""

# after trained, evaluate the performance of the trained model. 
for step in 25 50 75 100; do
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 python -u  /root/autodl-tmp/keshihua/eval/evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "aime25" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size 2 \
        --checkpoint_dir "$EXP_DIR/checkpoint-$step"
done

