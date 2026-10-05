# RGR
## Datasets

### Training Dataset

Following OPSD, we use the same publicly available dataset for training:

- **OpenThoughts-Math-30K-OPSD:** [`siyanzhao/Openthoughts_math_30k_opsd`](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd)

### Evaluation Datasets

We evaluate our model on the following mathematical reasoning benchmarks:

- **AIME 2024:** [`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
- **AIME 2025:** [`MathArena/aime_2025`](https://huggingface.co/datasets/MathArena/aime_2025)
- **HMMT February 2025:** [`MathArena/hmmt_feb_2025`](https://huggingface.co/datasets/MathArena/hmmt_feb_2025)

## Model Preparation

Download the Qwen3-1.7B and Qwen3-4B models from [ModelScope](https://modelscope.cn/).

## Running Experiments

Before running an experiment, please configure the model, dataset, checkpoint, and output paths according to your local environment.

### OPSD

From the repository root, configure the required paths in `OPSD/run_opsd_1b.sh`, and then run:

```bash
bash OPSD/run_opsd_1b.sh
```

### SDPO, SRPO, and RLSD

From the repository root, first enter the `SDPO+SRPO+RLSD` directory:

```bash
cd SDPO+SRPO+RLSD
```

Configure the required paths in the corresponding YAML file, and then run one of the following commands from this directory.

#### SDPO

```bash
bash scripts/_run_verl.sh \
  "$(pwd)/configs/qwen3_1_7b_sdpo.yaml" \
  2>&1 | tee sdpo_launcher.log
```

#### SRPO

```bash
bash scripts/_run_verl.sh \
  "$(pwd)/configs/qwen3_1_7b_srpo.yaml" \
  2>&1 | tee srpo_launcher.log
```

#### RLSD

```bash
bash scripts/_run_verl.sh \
  "$(pwd)/configs/qwen3_1_7b_rlsd.yaml" \
  2>&1 | tee rlsd_launcher.log
```

> **Note:** Before running an experiment, replace the model, dataset, checkpoint, and output paths in the corresponding launch script or configuration file with valid paths for your local environment.
