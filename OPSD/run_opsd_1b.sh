export CUDA_VISIBLE_DEVICES=0,1,2,3

unset WANDB_DISABLED
export WANDB_MODE=disabled

accelerate launch \
    --config_file  \
    --num_processes 4 \
    --gradient_accumulation_steps 2 \
    --main_process_port 12949 \
    /opsd_train.py \
    --model_name_or_path  \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 2 \
    --output_dir \
    --run_config  \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --bf16 true \
    --fp16 false \
    --tf32 false \
    --max_length 20000 \
    --beta 0 \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --jsd_token_clip 0.05 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --fixed_teacher \
    --use_hidden_penalty true\
    --hidden_penalty_weight 0.05 \
    --base_snapshot_steps 400 \
    --base_teacher_snapshot_steps 25,50,75 \
    --report_to none
