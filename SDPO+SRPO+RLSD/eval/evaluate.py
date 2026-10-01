import os
import json
import random
import argparse
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

seed = 42

os.environ["PYTHONHASHSEED"] = str(seed)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from math_verify import parse, verify


def extract_boxed_answer(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None

    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1

        if text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break

        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx:right_brace_idx + 1]

    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        answer = boxed_str[7:-1]
        return answer.strip()

    return None


def grade_answer(predicted, ground_truth) -> bool:
    if predicted is None or ground_truth is None:
        return False

    predicted = str(predicted).strip()
    ground_truth = str(ground_truth).strip()

    if not predicted or not ground_truth:
        return False

    try:
        if "$" not in predicted:
            predicted = f"${predicted}$"

        if "$" not in ground_truth:
            ground_truth = f"${ground_truth}$"

        pred_parsed = parse(
            predicted,
            fallback_mode="no_fallback",
        )
        gt_parsed = parse(
            ground_truth,
            fallback_mode="no_fallback",
        )

        return verify(
            gt_parsed,
            pred_parsed,
            timeout_seconds=5,
        )

    except Exception:
        pred_norm = (
            predicted.replace("$", "")
            .replace(" ", "")
            .lower()
            .strip()
        )
        gt_norm = (
            ground_truth.replace("$", "")
            .replace(" ", "")
            .lower()
            .strip()
        )

        return pred_norm == gt_norm


def has_lora_weights(path: Path) -> bool:
    return (
        (path / "adapter_model.safetensors").is_file()
        or (path / "adapter_model.bin").is_file()
    )


def resolve_checkpoint_layout(
    checkpoint_dir: Optional[str],
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    if checkpoint_dir is None:
        return None, None, None

    input_path = Path(checkpoint_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {input_path}"
        )

    lora_candidates = [
        input_path,
        input_path / "lora_adapter",
        input_path / "actor" / "lora_adapter",
    ]

    lora_adapter_path = None

    for candidate in lora_candidates:
        if candidate.is_dir() and has_lora_weights(candidate):
            lora_adapter_path = candidate
            break

    if lora_adapter_path is None:
        checked_paths = "\n".join(
            f"  - {candidate}" for candidate in lora_candidates
        )
        raise FileNotFoundError(
            "Could not find LoRA adapter weights.\n"
            "Expected adapter_model.safetensors or adapter_model.bin in:\n"
            f"{checked_paths}"
        )

    if lora_adapter_path.name == "lora_adapter":
        actor_path = lora_adapter_path.parent
    elif input_path.name == "actor":
        actor_path = input_path
    elif (input_path / "actor").is_dir():
        actor_path = input_path / "actor"
    else:
        actor_path = None

    tokenizer_path = None

    if actor_path is not None:
        hf_path = actor_path / "huggingface"

        if hf_path.is_dir() and (
            (hf_path / "tokenizer.json").is_file()
            or (hf_path / "tokenizer_config.json").is_file()
        ):
            tokenizer_path = hf_path

    global_step_path = None

    for parent in [input_path, *input_path.parents]:
        if parent.name.startswith("global_step_"):
            global_step_path = parent
            break

    return lora_adapter_path, tokenizer_path, global_step_path


def get_checkpoint_label(
    checkpoint_dir: Optional[str],
    global_step_path: Optional[Path],
) -> str:
    if global_step_path is not None:
        return global_step_path.name

    if checkpoint_dir:
        return Path(checkpoint_dir).name

    return "base_model"


def print_model_dtype_information(llm):
    print("\n" + "=" * 70)
    print("MODEL DTYPE INFORMATION")
    print("=" * 70)

    try:
        print(
            f"vLLM Model Config dtype: "
            f"{llm.llm_engine.model_config.dtype}"
        )
    except Exception:
        print("vLLM Model Config dtype: unavailable")

    try:
        print(
            f"vLLM Model quantization: "
            f"{llm.llm_engine.model_config.quantization}"
        )
    except Exception:
        print("vLLM Model quantization: unavailable")

    try:
        print(
            f"KV cache dtype: "
            f"{llm.llm_engine.cache_config.cache_dtype}"
        )
    except Exception:
        print("KV cache dtype: unavailable")

    print("=" * 70 + "\n")


def load_vllm_model(
    base_model_path: str,
    lora_adapter_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    enable_thinking: bool = True,
    max_lora_rank: Optional[int] = None,
):
    base_model_path = str(Path(base_model_path).expanduser())

    print(f"Loading base model with vLLM from: {base_model_path}")

    if max_model_len is None:
        max_model_len = 40960 if enable_thinking else 32768
        print(
            f"Auto-setting max_model_len to {max_model_len} for "
            f"{'thinking' if enable_thinking else 'non-thinking'} mode"
        )
    llm_config = {
        "model": base_model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "distributed_executor_backend": "mp",
        "enforce_eager": True,
    }

    if lora_adapter_path is not None:
        lora_adapter_path = Path(lora_adapter_path).resolve()

        if not has_lora_weights(lora_adapter_path):
            raise FileNotFoundError(
                f"No LoRA weights found in: {lora_adapter_path}"
            )

        if max_lora_rank is not None and max_lora_rank != 64:
            raise ValueError(
                "max_lora_rank must be 64 to match the first script."
            )

        print(f"LoRA adapter path: {lora_adapter_path}")
        print("Maximum LoRA rank: 64")
        print("Enabling vLLM LoRA support...")
        llm_config["enable_lora"] = True
        llm_config["max_lora_rank"] = 64
        llm_config["max_loras"] = 1
        llm_config["max_cpu_loras"] = 1

    llm = LLM(**llm_config)
    tokenizer_source = tokenizer_path or base_model_path

    print(f"Loading tokenizer from: {tokenizer_source}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )

    print_model_dtype_information(llm)
    print("vLLM model loaded successfully!")

    return llm, tokenizer


def load_local_dataset(dataset_path: str):
    
    path_str = os.path.expanduser(dataset_path)
    path = Path(path_str)

    if path.is_dir() and (
        (path / "dataset_info.json").exists()
        or (path / "state.json").exists()
        or (path / "dataset_dict.json").exists()
    ):
        dataset = load_from_disk(str(path))

        if hasattr(dataset, "keys"):
            if "test" in dataset:
                return dataset["test"]
            if "train" in dataset:
                return dataset["train"]

            first_split = next(iter(dataset.keys()))
            return dataset[first_split]

        return dataset

    lowercase_path = path_str.lower()

    if (
        lowercase_path.endswith(".json")
        or lowercase_path.endswith(".jsonl")
        or ".json*" in lowercase_path
    ):
        return load_dataset(
            "json",
            data_files=path_str,
            split="train",
        )

    return load_dataset(
        "parquet",
        data_files=path_str,
        split="train",
    )


def load_evaluation_dataset(
    dataset_name: str,
    dataset_path: Optional[str] = None,
):
    dataset_name = dataset_name.lower()

    if dataset_path:
        print(f"Loading local dataset from: {dataset_path}")
        dataset = load_local_dataset(dataset_path)
        print(
            f"Loaded local {dataset_name.upper()} dataset "
            f"with {len(dataset)} problems"
        )
        return dataset

    if dataset_name == "math500":
        dataset = load_dataset(
            "MATH-500",
            split="test",
        )

    elif dataset_name == "amo-bench":
        dataset = load_dataset(
            "AMO-Bench",
            split="test",
        )

    elif dataset_name == "minerva":
        dataset = load_dataset(
            "minervamath",
            split="test",
        )

    elif dataset_name == "amc23":
        dataset = load_dataset(
            "amc23",
            split="test",
        )

    elif dataset_name in {"aime24", "aime25", "hmmt25"}:
        raise ValueError(
            f"Dataset '{dataset_name}' uses a local dataset in your "
            f"original code, but --dataset_path was not provided.\n"
            f"Example:\n"
            f"  --dataset {dataset_name} "
            f"--dataset_path '/path/to/train-*.parquet'"
        )

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from: "
            f"math500, amo-bench, aime24, aime25, hmmt25, "
            f"minerva, amc23"
        )

    print(
        f"Loaded {dataset_name.upper()} dataset "
        f"with {len(dataset)} problems"
    )

    return dataset
def parse_dataset_example(example, dataset_name: str):
    dataset_name = dataset_name.lower()

    if dataset_name == "amo-bench":
        problem = example["prompt"]
        gt_answer = example["answer"]
        question_id = example.get("question_id", None)

    elif dataset_name == "aime24":
        problem = example["problem"]
        gt_answer = example["answer"]
        question_id = example.get("id", None)

    elif dataset_name == "minerva":
        problem = example["question"]
        gt_answer = example["answer"]
        question_id = example.get("id", None)

    elif dataset_name == "amc23":
        problem = example["question"]
        gt_answer = example["answer"]
        question_id = example.get("id", None)

    elif dataset_name == "aime25":
        problem = example["problem"]
        gt_answer = example["answer"]
        question_id = example.get("id", None)

    elif dataset_name == "hmmt25":
        problem = example["problem"]
        gt_answer = str(example["answer"])
        question_id = example.get("problem_idx", None)

    else:
        # MATH500：严格按第一段从 solution 提取答案。
        problem = example["problem"]
        gt_solution = example["solution"]
        question_id = None

        gt_answer = extract_boxed_answer(gt_solution)
        if gt_answer is None:
            gt_answer = gt_solution

    return problem, gt_answer, question_id
def evaluate_math500(
    llm,
    tokenizer,
    max_new_tokens: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    num_samples: Optional[int] = None,
    output_file: Optional[str] = None,
    lora_request=None,
    dataset_name: str = "math500",
    dataset_path: Optional[str] = None,
    base_model_name: Optional[str] = None,
    enable_thinking: bool = True,
    val_n: int = 1,
):
    print(f"\n{'=' * 70}")
    print("EVALUATION CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Dataset path: {dataset_path or 'Hugging Face/default'}")
    print(
        f"Thinking Mode: "
        f"{'ENABLED' if enable_thinking else 'DISABLED'}"
    )
    print(f"Temperature: {temperature}")
    print(f"Top-P: {top_p}")
    print(f"Top-K: {top_k}")
    print(f"Min-P: {min_p}")
    print(f"Presence Penalty: {presence_penalty}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Val-N (solutions per problem): {val_n}")
    print(f"{'=' * 70}\n")
    dataset = load_evaluation_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
    )
    if num_samples:
        dataset = dataset.select(
            range(min(num_samples, len(dataset)))
        )

    print(
        f"Evaluating on {len(dataset)} problems "
        f"with vLLM batch inference..."
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_new_tokens,
        presence_penalty=presence_penalty,
        n=val_n,
    )

    total = 0
    formatted_count = 0
    results = []

    pass_at_n = 0
    total_correct_per_problem = 0

    all_messages = []
    all_gt_answers = []
    all_problems = []
    all_question_ids = []

    for example in dataset:
        problem, gt_answer, question_id = parse_dataset_example(
            example,
            dataset_name,
        )
        user_message = (
            f"{problem}\n\nPlease reason step by step, "
            f"and put your final answer within \\boxed{{}}."
        )

        messages = [
            {
                "role": "user",
                "content": user_message,
            }
        ]

        all_messages.append(messages)
        all_gt_answers.append(gt_answer)
        all_problems.append(problem)
        all_question_ids.append(question_id)

    print(
        f"\nRunning vLLM batch inference on "
        f"{len(all_messages)} problems..."
    )
    print("Using generate interface with manual chat template...")

    all_prompts = []

    for messages in all_messages:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        all_prompts.append(text)

    print_model_dtype_information(llm)
    print(f"Using LoRA: {lora_request is not None}")

    if lora_request is not None:
        lora_path = getattr(
            lora_request,
            "lora_path",
            getattr(lora_request, "lora_local_path", None),
        )

        if lora_path is None:
            raise ValueError(
                "LoRA request created but LoRA path is None."
            )

        print(f"LoRA path: {lora_path}")
    if lora_request is not None:
        outputs = llm.generate(
            all_prompts,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=True,
        )
    else:
        outputs = llm.generate(
            all_prompts,
            sampling_params,
            use_tqdm=True,
        )

    print("\nProcessing results...")

    for idx, (output, problem, gt_answer, question_id) in enumerate(
        zip(
            outputs,
            all_problems,
            all_gt_answers,
            all_question_ids,
        )
    ):
        generations = []
        predicted_answers = []
        is_correct_list = []
        is_formatted_list = []

        for i in range(len(output.outputs)):
            generated_text = output.outputs[i].text

            predicted_answer = extract_boxed_answer(generated_text)
            is_formatted = predicted_answer is not None
            is_correct = grade_answer(
                predicted_answer,
                gt_answer,
            )

            generations.append(generated_text)

            predicted_answers.append(
                predicted_answer
                if predicted_answer
                else "[No boxed answer found]"
            )

            is_correct_list.append(is_correct)
            is_formatted_list.append(is_formatted)
        num_correct = sum(is_correct_list)
        num_formatted = sum(is_formatted_list)
        has_correct = any(is_correct_list)

        majority_vote_correct = False

        if num_formatted > 0:
            formatted_predictions = [
                pred
                for pred, fmt in zip(
                    predicted_answers,
                    is_formatted_list,
                )
                if fmt
            ]

            if formatted_predictions:
                most_common_answer = Counter(
                    formatted_predictions
                ).most_common(1)[0][0]

                majority_vote_correct = grade_answer(
                    most_common_answer,
                    gt_answer,
                )

        if has_correct:
            pass_at_n += 1

        total_correct_per_problem += num_correct
        formatted_count += num_formatted
        total += val_n
        result = {
            "problem_id": (
                question_id if question_id is not None else idx
            ),
            "problem": problem,
            "ground_truth": gt_answer,
            "val_n": val_n,
            "generations": [
                {
                    "predicted_answer": pred,
                    "full_generation": gen,
                    "correct": corr,
                    "formatted": fmt,
                }
                for pred, gen, corr, fmt in zip(
                    predicted_answers,
                    generations,
                    is_correct_list,
                    is_formatted_list,
                )
            ],
            "num_correct": num_correct,
            "pass_at_n": has_correct,
            "majority_vote_correct": majority_vote_correct,
            "predicted_answer": predicted_answers[0],
            "full_generation": generations[0],
            "correct": is_correct_list[0],
            "formatted": is_formatted_list[0],
        }

        results.append(result)

        format_rate = formatted_count / total * 100
        current_pass_at_n = pass_at_n / (idx + 1) * 100
        current_avg_at_n = total_correct_per_problem / total * 100

        status = "✓" if has_correct else "✗"

        print(
            f"{status} [{idx + 1}/{len(dataset)}] "
            f"Pass@{val_n}: {current_pass_at_n:.1f}% | "
            f"Avg@{val_n}: {current_avg_at_n:.1f}% | "
            f"Formatted: {format_rate:.1f}%"
        )

        if (idx + 1) % 10 == 0:
            print(f"\n{'=' * 70}")
            print(f"Progress: {idx + 1}/{len(dataset)}")
            print(f"Pass@{val_n}: {current_pass_at_n:.2f}%")
            print(f"Average@{val_n}: {current_avg_at_n:.2f}%")
            print(f"Format Rate: {format_rate:.2f}%")
            print(f"Last problem: {problem[:100]}...")
            print(f"Solutions correct: {num_correct}/{val_n}")
            print(
                f"Majority vote: "
                f"{'✓' if majority_vote_correct else '✗'}"
            )
            print(f"Ground truth: {gt_answer}")
            print(f"{'=' * 70}\n")

    num_problems = len(dataset)

    format_rate = formatted_count / total * 100
    pass_at_n_pct = pass_at_n / num_problems * 100
    average_at_n_pct = total_correct_per_problem / total * 100

    majority_vote_correct_count = sum(
        1 for result in results
        if result["majority_vote_correct"]
    )

    majority_vote_at_n_pct = (
        majority_vote_correct_count / num_problems * 100
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Dataset: {dataset_name.upper()}")
    print(
        f"Thinking Mode: "
        f"{'ENABLED' if enable_thinking else 'DISABLED'}"
    )
    print(f"Total problems: {num_problems}")
    print(f"Solutions per problem: {val_n}")
    print(f"Total solutions: {total}")

    print("\nMetrics:")
    print(
        f"  Pass@{val_n}: {pass_at_n_pct:.2f}% "
        f"({pass_at_n}/{num_problems})"
    )
    print(
        f"  Average@{val_n}: {average_at_n_pct:.2f}% "
        f"({total_correct_per_problem}/{total})"
    )
    print(
        f"  Majority Vote@{val_n}: "
        f"{majority_vote_at_n_pct:.2f}% "
        f"({majority_vote_correct_count}/{num_problems})"
    )

    print("\nFormatting:")
    print(f"  Formatted (boxed) answers: {formatted_count}/{total}")
    print(f"  Format rate: {format_rate:.2f}%")
    print("=" * 70)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary = {
            "base_model": base_model_name,
            "dataset": dataset_name,
            "enable_thinking": enable_thinking,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_new_tokens": max_new_tokens,
            "val_n": val_n,
            "num_problems": num_problems,
            "total_solutions": total,
            "pass_at_n": pass_at_n,
            "pass_at_n_pct": pass_at_n_pct,
            "average_at_n": total_correct_per_problem,
            "average_at_n_pct": average_at_n_pct,
            "majority_vote_at_n": majority_vote_correct_count,
            "majority_vote_at_n_pct": majority_vote_at_n_pct,
            "formatted_count": formatted_count,
            "format_rate": format_rate,
            "results": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                summary,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\nDetailed results saved to: {output_file}")

    return average_at_n_pct, results



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Use the second script's file selection and paths, "
            "with the first script's evaluation logic."
        )
    )

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help=(
            "Original base model path. It must match the model "
            "used to train the LoRA adapter."
        ),
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "Checkpoint path: global_step_X, global_step_X/actor, "
            "actor/lora_adapter, or an old adapter directory."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="math500",
        choices=[
            "math500",
            "amo-bench",
            "aime24",
            "aime25",
            "hmmt25",
            "minerva",
            "amc23",
        ],
        help="Dataset to evaluate.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help=(
            "Optional local dataset directory or parquet/json/jsonl "
            "path/glob. Required for local AIME/HMMT datasets."
        ),
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=38912,
        help="Maximum generated tokens.",
    )

    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=True,
        help="Enable Qwen3 thinking mode.",
    )

    parser.add_argument(
        "--no_thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen3 thinking mode.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Auto: 0.95 for thinking, 0.8 for non-thinking.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k; -1 disables top-k filtering.",
    )

    parser.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="Minimum probability threshold.",
    )

    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Presence penalty.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate; None means all.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Detailed result JSON path.",
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization.",
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs used by tensor parallelism.",
    )

    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Auto: 40960 for thinking, 32768 for non-thinking.",
    )

    parser.add_argument(
        "--val_n",
        type=int,
        default=6,
        help="Number of generations for each problem.",
    )

    # 保留参数接口，但限定为第一段使用的值。
    parser.add_argument(
        "--max_lora_rank",
        type=int,
        default=64,
        choices=[64],
        help="Fixed at 64 to match the first script.",
    )

    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help=(
            "Optional tokenizer path. By default the program uses "
            "checkpoint/actor/huggingface when available."
        ),
    )

    args = parser.parse_args()

    if args.val_n <= 0:
        raise ValueError("--val_n must be greater than zero.")

    if args.tensor_parallel_size <= 0:
        raise ValueError(
            "--tensor_parallel_size must be greater than zero."
        )

    if not Path(args.base_model).expanduser().exists():
        print(
            f"Warning: base model path does not exist locally: "
            f"{args.base_model}"
        )
        print(
            "If this is a Hugging Face model ID, this warning "
            "can be ignored."
        )

    lora_adapter_path = None
    checkpoint_tokenizer_path = None
    global_step_path = None

    if args.checkpoint_dir is not None:
        (
            lora_adapter_path,
            checkpoint_tokenizer_path,
            global_step_path,
        ) = resolve_checkpoint_layout(args.checkpoint_dir)

        print("\n" + "=" * 70)
        print("CHECKPOINT LAYOUT")
        print("=" * 70)
        print(f"Input checkpoint: {args.checkpoint_dir}")
        print(f"Global step dir: {global_step_path}")
        print(f"LoRA adapter: {lora_adapter_path}")
        print(f"Checkpoint tokenizer: {checkpoint_tokenizer_path}")
        print("=" * 70 + "\n")
    tokenizer_path = (
        args.tokenizer_path
        or (
            str(checkpoint_tokenizer_path)
            if checkpoint_tokenizer_path is not None
            else None
        )
    )

    checkpoint_label = get_checkpoint_label(
        args.checkpoint_dir,
        global_step_path,
    )

    if args.top_p is None:
        args.top_p = 0.95 if args.enable_thinking else 0.8
        print(
            f"Auto-setting top_p={args.top_p} for "
            f"{'thinking' if args.enable_thinking else 'non-thinking'} "
            f"mode"
        )

    if args.enable_thinking and args.temperature == 0.0:
        print("\n" + "!" * 70)
        print(
            "WARNING: Using greedy decoding "
            "(temperature=0.0) in thinking mode!"
        )
        print("!" * 70 + "\n")
    if args.output_file is None:
        output_parts = [
            "eval",
            args.dataset,
            Path(args.base_model).name,
            checkpoint_label,
            (
                "thinking"
                if args.enable_thinking
                else "nonthinking"
            ),
            f"temp{args.temperature}",
            f"valn{args.val_n}",
        ]

        args.output_file = str(
            Path("eval_results_1")
            / ("_".join(output_parts) + ".json")
        )

    print(f"Results will be saved to: {args.output_file}")

    print("\n" + "=" * 70)
    print("QWEN3 MATH EVALUATION")
    print("=" * 70)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint input: {args.checkpoint_dir}")
    print(f"Resolved LoRA: {lora_adapter_path}")
    print(f"Tokenizer: {tokenizer_path or args.base_model}")
    print(
        f"Thinking mode: "
        f"{'ENABLED' if args.enable_thinking else 'DISABLED'}"
    )
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Min-p: {args.min_p}")
    print(f"Presence penalty: {args.presence_penalty}")
    print(f"Num samples: {args.num_samples or 'All'}")
    print(f"Val-N: {args.val_n}")
    print(f"Output file: {args.output_file}")
    print(
        f"GPU memory utilization: "
        f"{args.gpu_memory_utilization}"
    )
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("Distributed executor backend: mp")
    print(f"Max LoRA rank: {args.max_lora_rank}")
    print("=" * 70 + "\n")

    llm, tokenizer = load_vllm_model(
        base_model_path=args.base_model,
        lora_adapter_path=(
            str(lora_adapter_path)
            if lora_adapter_path is not None
            else None
        ),
        tokenizer_path=tokenizer_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
        max_lora_rank=args.max_lora_rank,
    )
    lora_request = None

    if lora_adapter_path is not None:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest(
            "checkpoint_lora",
            1,
            str(lora_adapter_path),
        )

        print(
            f"✓ Successfully created LoRA request: "
            f"{lora_adapter_path}"
        )

    average_at_n_pct, results = evaluate_math500(
        llm=llm,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        num_samples=args.num_samples,
        output_file=args.output_file,
        lora_request=lora_request,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        base_model_name=args.base_model,
        enable_thinking=args.enable_thinking,
        val_n=args.val_n,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Final Average@{args.val_n}: {average_at_n_pct:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
