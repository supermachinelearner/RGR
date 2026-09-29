#!/bin/bash

set -u
set -o pipefail


BASE_MODEL=""

EXP_DIR=""
EVAL_PY=""

OUTPUT_ROOT="${EXP_DIR}/eval_results_1"


CUDA_DEVICES="0,1,2,3"
TP_SIZE=4

GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=40960
MAX_NEW_TOKENS=38912
VAL_N=12

TEMPERATURE=1.0
TOP_P=0.95
TOP_K=-1
MIN_P=0.0
PRESENCE_PENALTY=0.0
MAX_LORA_RANK=64


# ============================================================
# Steps
# ============================================================

STEPS=(
    25
    50
    75
    100
    125
)


DATASETS=(
    aime24
    aime25
    hmmt25
)

declare -A DATASET_PATHS

DATASET_PATHS[aime24]=
DATASET_PATHS[aime25]=
DATASET_PATHS[hmmt25]=

DATASET_PATHS[math500]=""
DATASET_PATHS[minerva]=""
DATASET_PATHS[amc23]=""
DATASET_PATHS[amo-bench]=""


NUM_SAMPLES=""

STOP_ON_ERROR=0

SKIP_EXISTING=0


if ! command -v python > /dev/null 2>&1; then
    echo "[ERROR] python not found. Please activate your environment."
    exit 1
fi

if [[ ! -f "${EVAL_PY}" ]]; then
    echo "[ERROR] Evaluation script not found:"
    echo "        ${EVAL_PY}"
    exit 1
fi

if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "[ERROR] Base model directory not found:"
    echo "        ${BASE_MODEL}"
    exit 1
fi

if [[ ! -d "${EXP_DIR}" ]]; then
    echo "[ERROR] Experiment directory not found:"
    echo "        ${EXP_DIR}"
    exit 1
fi

if ! mkdir -p "${OUTPUT_ROOT}"; then
    echo "[ERROR] Cannot create output directory:"
    echo "        ${OUTPUT_ROOT}"
    exit 1
fi

echo "============================================================"
echo "Multi-step multi-dataset evaluation"
echo "============================================================"
echo "Python      : $(command -v python)"
echo "Base model  : ${BASE_MODEL}"
echo "Experiment  : ${EXP_DIR}"
echo "Eval script : ${EVAL_PY}"
echo "GPUs        : ${CUDA_DEVICES}"
echo "TP size     : ${TP_SIZE}"
echo "Steps       : ${STEPS[*]}"
echo "Datasets    : ${DATASETS[*]}"
echo "Val-N       : ${VAL_N}"
echo "Temperature : ${TEMPERATURE}"
echo "Top-P       : ${TOP_P}"
echo "Top-K       : ${TOP_K}"
echo "LoRA rank   : ${MAX_LORA_RANK}"
echo "Skip old    : ${SKIP_EXISTING}"
echo "Output      : ${OUTPUT_ROOT}"
echo "============================================================"


TOTAL_TASKS=0
SUCCESS_TASKS=0
FAILED_TASKS=0
SKIPPED_TASKS=0

for step in "${STEPS[@]}"; do
    CHECKPOINT_DIR="${EXP_DIR}/global_step_${step}"
    LORA_DIR="${CHECKPOINT_DIR}/actor/lora_adapter"

    for dataset in "${DATASETS[@]}"; do
        TOTAL_TASKS=$((TOTAL_TASKS + 1))

        DATASET_PATH="${DATASET_PATHS[$dataset]:-}"

        STEP_OUTPUT_DIR="${OUTPUT_ROOT}/global_step_${step}"

        OUTPUT_FILE="${STEP_OUTPUT_DIR}/${dataset}_thinking_temp${TEMPERATURE}_valn${VAL_N}.json"
        LOG_FILE="${STEP_OUTPUT_DIR}/${dataset}_thinking_temp${TEMPERATURE}_valn${VAL_N}.log"

        echo
        echo "============================================================"
        echo "Evaluation task"
        echo "============================================================"
        echo "Step          : ${step}"
        echo "Dataset       : ${dataset}"
        echo "Checkpoint    : ${CHECKPOINT_DIR}"
        echo "LoRA          : ${LORA_DIR}"
        echo "Dataset path  : ${DATASET_PATH:-Hugging Face/default}"
        echo "Output        : ${OUTPUT_FILE}"
        echo "Log           : ${LOG_FILE}"
        echo "============================================================"

        if [[ "${SKIP_EXISTING}" -eq 1 ]] && \
           [[ -s "${OUTPUT_FILE}" ]]; then
            echo "[SKIP] Existing output found: ${OUTPUT_FILE}"
            SKIPPED_TASKS=$((SKIPPED_TASKS + 1))
            continue
        fi

        TASK_ERROR=""

        if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
            TASK_ERROR="Checkpoint directory not found: ${CHECKPOINT_DIR}"

        elif [[ ! -f "${LORA_DIR}/adapter_model.safetensors" ]] && \
             [[ ! -f "${LORA_DIR}/adapter_model.bin" ]]; then
            TASK_ERROR="LoRA weights not found: ${LORA_DIR}"

        elif [[ ! -f "${LORA_DIR}/adapter_config.json" ]]; then
            TASK_ERROR="LoRA config not found: ${LORA_DIR}/adapter_config.json"
        fi

        if [[ -z "${TASK_ERROR}" ]] && [[ -n "${DATASET_PATH}" ]]; then
            if [[ ! -e "${DATASET_PATH}" ]] && \
               ! compgen -G "${DATASET_PATH}" > /dev/null; then
                TASK_ERROR="Dataset path not found: ${DATASET_PATH}"
            fi
        fi


        if [[ -z "${TASK_ERROR}" ]] && [[ -z "${DATASET_PATH}" ]]; then
            case "${dataset}" in
                aime24|aime25|hmmt25)
                    TASK_ERROR="Local dataset path is required for ${dataset}"
                    ;;
            esac
        fi

        if [[ -z "${TASK_ERROR}" ]]; then
            if ! mkdir -p "${STEP_OUTPUT_DIR}"; then
                TASK_ERROR="Cannot create output directory: ${STEP_OUTPUT_DIR}"
            fi
        fi

        if [[ -n "${TASK_ERROR}" ]]; then
            echo "[ERROR] ${TASK_ERROR}"
            FAILED_TASKS=$((FAILED_TASKS + 1))

            if [[ "${STOP_ON_ERROR}" -eq 1 ]]; then
                exit 1
            fi

            continue
        fi


        CMD=(
            python -u "${EVAL_PY}"

            --base_model "${BASE_MODEL}"
            --checkpoint_dir "${CHECKPOINT_DIR}"

            --dataset "${dataset}"

            --val_n "${VAL_N}"
            --temperature "${TEMPERATURE}"
            --top_p "${TOP_P}"
            --top_k "${TOP_K}"
            --min_p "${MIN_P}"
            --presence_penalty "${PRESENCE_PENALTY}"

            --max_new_tokens "${MAX_NEW_TOKENS}"
            --max_model_len "${MAX_MODEL_LEN}"

            --tensor_parallel_size "${TP_SIZE}"
            --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
            --max_lora_rank "${MAX_LORA_RANK}"

            --output_file "${OUTPUT_FILE}"
            --enable_thinking
        )

        if [[ -n "${DATASET_PATH}" ]]; then

            CMD+=(
                --dataset_path "${DATASET_PATH}"
            )
        fi

        if [[ -n "${NUM_SAMPLES}" ]]; then
            CMD+=(
                --num_samples "${NUM_SAMPLES}"
            )
        fi


        echo
        echo "Command:"
        printf " %q" "${CMD[@]}"
        echo
        echo


        NCCL_P2P_DISABLE=1 \
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
        TOKENIZERS_PARALLELISM=false \
        "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
        PIPE_CODES=("${PIPESTATUS[@]}")
        PYTHON_EXIT_CODE="${PIPE_CODES[0]}"
        TEE_EXIT_CODE="${PIPE_CODES[1]}"

        TASK_EXIT_CODE=0

        if [[ "${PYTHON_EXIT_CODE}" -ne 0 ]]; then
            echo
            echo "[ERROR] Python exited with code ${PYTHON_EXIT_CODE}"
            TASK_EXIT_CODE="${PYTHON_EXIT_CODE}"

        elif [[ "${TEE_EXIT_CODE}" -ne 0 ]]; then
            echo
            echo "[ERROR] Log writing failed with code ${TEE_EXIT_CODE}"
            TASK_EXIT_CODE="${TEE_EXIT_CODE}"

        elif [[ ! -s "${OUTPUT_FILE}" ]]; then
            echo
            echo "[ERROR] Python exited successfully, but output is missing or empty:"
            echo "        ${OUTPUT_FILE}"
            TASK_EXIT_CODE=1
        fi

        if [[ "${TASK_EXIT_CODE}" -eq 0 ]]; then
            echo
            echo "[SUCCESS] step=${step}, dataset=${dataset}"
            SUCCESS_TASKS=$((SUCCESS_TASKS + 1))
        else
            echo
            echo "[FAILED] step=${step}, dataset=${dataset}, exit=${TASK_EXIT_CODE}"
            FAILED_TASKS=$((FAILED_TASKS + 1))

            if [[ "${STOP_ON_ERROR}" -eq 1 ]]; then
                echo "STOP_ON_ERROR=1, stopping."
                exit "${TASK_EXIT_CODE}"
            fi
        fi
    done
done


# ============================================================
# Summary
# ============================================================

echo
echo "============================================================"
echo "All evaluation tasks finished"
echo "============================================================"
echo "Total tasks   : ${TOTAL_TASKS}"
echo "Successful    : ${SUCCESS_TASKS}"
echo "Failed        : ${FAILED_TASKS}"
echo "Skipped       : ${SKIPPED_TASKS}"
echo "Results dir   : ${OUTPUT_ROOT}"
echo "============================================================"

if [[ "${FAILED_TASKS}" -gt 0 ]]; then
    exit 1
fi

exit 0