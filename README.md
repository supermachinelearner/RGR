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


### FlashAttention Installation

> [!IMPORTANT]
> FlashAttention is sensitive to the Python, PyTorch, CUDA, C++ ABI, operating system, and system architecture configurations. Installing an incompatible wheel may result in installation failures or import errors.

Our experiments were conducted using the following environment:

- Python 3.11
- PyTorch 2.8.0
- CUDA 12.8
- FlashAttention 2.8.3
- Linux x86_64
- CXX11 ABI: `TRUE`

For our environment, we use the following pre-built FlashAttention wheel:

```text
flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

The wheel can be downloaded from the official FlashAttention v2.8.3 release page:

https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3

After downloading the wheel, install it using:

```bash
python -m pip install "/path/to/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
```

Replace `/path/to/` with the actual path to the downloaded wheel.

Before installation, run the following command to check your environment:

```bash
python - <<'PY'
import platform
import sys

print("Environment information:")
print(f"  Python:       {platform.python_version()}")
print(f"  Python tag:   cp{sys.version_info.major}{sys.version_info.minor}")
print(f"  OS:           {platform.system()}")
print(f"  Architecture: {platform.machine()}")

try:
    import torch

    print(f"  PyTorch:      {torch.__version__}")
    print(f"  PyTorch CUDA: {torch.version.cuda or 'Unavailable'}")
    print(f"  CUDA usable:  {torch.cuda.is_available()}")
    print(f"  CXX11 ABI:    {torch._C._GLIBCXX_USE_CXX11_ABI}")

    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")

except ImportError:
    print("  PyTorch:      Not installed")
    print(
        "\n[WARNING] PyTorch is not installed. Please install PyTorch "
        "before installing FlashAttention."
    )
PY
```

Select a FlashAttention wheel that matches the detected Python version, PyTorch version, CUDA version, CXX11 ABI setting, operating system, and system architecture.
