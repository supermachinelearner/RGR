import os
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig
import os
import random
import numpy as np
import torch

seed = 1024

os.environ["PYTHONHASHSEED"] = str(seed)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer import OPSDTrainer
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    student_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the student during rollout. "
            "Default False (matches the main OPSD setup: student rolls out without <think>)."
        },
    )
    teacher_thinking: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the teacher when scoring student tokens. "
            "Default True. Set to False for the matched non-thinking ablation (both nonthink)."
        },

    )
    use_hidden_penalty: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to add the Student/Base token-Gram penalty. "
                "The Base Model is initialized from the same checkpoint "
                "as the Student and remains frozen."
            )
        },
    )
    hidden_penalty_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Lambda coefficient for the Student/Base Gram loss. "
                "Final loss = distillation_loss + lambda * "
                "[MSE(Gram(H_student), Gram(H_base)) + "
                "||Gram(H_student) - Gram(H_base)||_F]."
            )
        },
    )
    base_snapshot_steps: str = field(
        default="25,50,75",
        metadata={
            "help": (
                "Optimizer steps at which the frozen Base snapshot "
                "is updated from the current Student LoRA weights. "
                "Example: 25,50,75"
            )
        },
    )
    base_teacher_snapshot_steps: str = field(
        default="25,50,75",
        metadata={
            "help": (
                "Optimizer steps at which the frozen Base Teacher snapshot "
                "is updated from the current Student LoRA weights. "
                "Example: 25,50,75"
            )
        },
    )

if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    base_snapshot_steps = tuple(
        sorted(
            {
                int(step.strip())
                for step in script_args.base_snapshot_steps.split(",")
                if step.strip()
            }
        )
    )
    base_teacher_snapshot_steps = tuple(
        sorted(
            {
                int(step.strip())
                for step in script_args.base_teacher_snapshot_steps.split(",")
                if step.strip()
            }
        )
    )
    if any(step <= 0 for step in base_snapshot_steps):
        raise ValueError(
            "base_snapshot_steps 中的 step 必须全部大于 0。"
        )
    if any(step <= 0 for step in base_snapshot_steps):
        raise ValueError(
            "base_snapshot_steps 中的 step 必须全部大于 0。"
        )
    if (
        script_args.use_hidden_penalty
        and script_args.hidden_penalty_weight <= 0
    ):
        raise ValueError(
            "use_hidden_penalty=True 时，"
            "hidden_penalty_weight 必须大于 0。"
        )
    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path (e.g., "")
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_config = (
            f"opsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to wandb name if enabled
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"
    if script_args.use_hidden_penalty:
        full_wandb_run_config += (
            f"_gram_mse_frob"
            f"{script_args.hidden_penalty_weight:g}"
        )

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Base Snapshot Steps: {base_snapshot_steps}")
    print(
        f"Base Teacher Snapshot Steps: "
        f"{base_teacher_snapshot_steps}"
    )
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                "use_hidden_penalty": script_args.use_hidden_penalty,
                "hidden_penalty_weight": (
                    script_args.hidden_penalty_weight
                    if script_args.use_hidden_penalty
                    else 0.0
                ),
                "hidden_penalty_type": (
                    "gram_mse_plus_frobenius"
                    if script_args.use_hidden_penalty
                    else None
                ),
                "base_model_initial_checkpoint": (
                    model_args.model_name_or_path
                    if script_args.use_hidden_penalty
                    else None
                ),
                "base_snapshot_steps": (
                    list(base_snapshot_steps)
                    if script_args.use_hidden_penalty
                    else None
                ),
                "base_teacher_snapshot_steps": (
                    list(base_teacher_snapshot_steps)
                    if script_args.use_hidden_penalty
                    else None
                ),
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch


    def normalize_dtype(value):
        if value is None:
            return None

        if isinstance(value, torch.dtype):
            return value

        if isinstance(value, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "float": torch.float32,
            }

            dtype = dtype_map.get(value.lower())
            if dtype is None:
                raise ValueError(f"Unsupported dtype: {value}")

            return dtype

        raise TypeError(f"Unsupported dtype type: {type(value)}")


    # ============================================================
    # 排障模式：强制 FP32 + eager
    # ============================================================
    FORCE_FP32_DEBUG = False

    if FORCE_FP32_DEBUG:
        model_dtype = torch.float32
        attention_implementation = "eager"

        # 防止 Trainer 使用混合精度
        training_args.fp16 = False
        training_args.bf16 = False

        if hasattr(training_args, "tf32"):
            training_args.tf32 = False

        # 使用完整 FP32 matmul，关闭 TF32
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    else:
        raw_dtype = None

        if (
            hasattr(model_args, "torch_dtype")
            and model_args.torch_dtype is not None
        ):
            raw_dtype = model_args.torch_dtype
        elif (
            hasattr(model_args, "dtype")
            and model_args.dtype is not None
        ):
            raw_dtype = model_args.dtype

        model_dtype = normalize_dtype(raw_dtype) or torch.bfloat16
        attention_implementation = (
            model_args.attn_implementation or "sdpa"
        )


    print(f"\n{'=' * 80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {attention_implementation}")
    print(f"Training fp16: {getattr(training_args, 'fp16', None)}")
    print(f"Training bf16: {getattr(training_args, 'bf16', None)}")
    print(f"Training tf32: {getattr(training_args, 'tf32', None)}")
    print(f"{'=' * 80}\n")


    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=attention_implementation,
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )


    quantization_config = get_quantization_config(model_args)

    if FORCE_FP32_DEBUG and quantization_config is not None:
        raise ValueError(
            "FP32 排障模式下不能启用 4-bit/8-bit 量化。"
            "请关闭 load_in_4bit/load_in_8bit 或其他量化配置。"
        )

    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config


    training_args.model_init_kwargs = model_kwargs

    # No separate teacher model needed - we use the same model with privileged info

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    dataset = load_dataset("")
    train_dataset = dataset["train"]

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),

        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=(
            script_args.top_k_loss
            if script_args.top_k_loss > 0
            else None
        ),
        jsd_token_clip=(
            script_args.jsd_token_clip
            if script_args.jsd_token_clip > 0
            else None
        ),
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        use_hidden_penalty=script_args.use_hidden_penalty,
        hidden_penalty_weight=script_args.hidden_penalty_weight,
        base_snapshot_steps=base_snapshot_steps,
        base_teacher_snapshot_steps=base_teacher_snapshot_steps,
    )
    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train(resume_from_checkpoint="")

    trainer.save_model(training_args.output_dir)
