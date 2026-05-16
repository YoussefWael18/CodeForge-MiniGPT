# CodeForge (Mini GPT)

A local fine-tuning project for `Qwen/Qwen2.5-Coder-1.5B-Instruct` using QLoRA. The goal is to train a small coding assistant that generates concise raw Python functions, compare it against the base model, and expose the merged fine-tuned model through a Streamlit interface.

---

## Project Overview

- Data preparation from local CodeSearchNet-style Parquet files
- QLoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct
- Export of a merged standalone model for local inference
- Evaluation of the base model versus the fine-tuned V1 model
- A Streamlit GUI for interactive code generation

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YoussefWael18/CodeForge-MiniGPT.git
cd CodeForge-MiniGPT
```

### 2. Install dependencies

```bash
pip install torch transformers datasets peft trl bitsandbytes streamlit safetensors accelerate
```

> GPU support is strongly recommended for training. The project is designed around 6GB VRAM using 4-bit quantization.

### 3. Train the model

Run `training.ipynb` from top to bottom to produce the merged model locally. See the Training section below for details.

---

## Repository Structure

```text
.
├── data preprocessing/
│   ├── data_prep.py
│   └── Processed_dataset/
│       └── golden_train.jsonl
├── Evaluation_results/
│   ├── evaluation.ipynb
│   └── evaluation_results.json
├── gui/
│   ├── app.py
│   └── README.md
├── training.ipynb
├── project_documentation.docx
└── README.md
```

---

## Data Preparation

The data preparation script loads local Parquet shards from `codesearchnet/pair/`, filters examples by instruction and code length, formats them in a ChatML-style prompt/response structure, and creates a curated JSONL training set.

```bash
python "data preprocessing/data_prep.py"
```

Output is saved to:

```text
data preprocessing/Processed_dataset/golden_train.jsonl
```

---

## Training

Open and run `training.ipynb` from top to bottom. The notebook:

1. Loads the processed JSONL dataset
2. Loads `Qwen/Qwen2.5-Coder-1.5B-Instruct` in 4-bit quantization
3. Configures and applies LoRA adapters
4. Runs supervised fine-tuning
5. Saves LoRA adapters
6. Merges adapters into a standalone model
7. Saves the merged model to `mini-gpt-coder-merged/`

### Key Training Settings

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-Coder-1.5B-Instruct |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | 2e-4 |
| Max sequence length | 2048 |
| Max steps | 300 |
| Precision | bfloat16 |
| Optimizer | paged_adamw_32bit |

---

## Running the GUI

```bash
streamlit run gui/app.py
```

Make sure `mini-gpt-coder-merged/` exists in the project root before running. The app provides controls for max new tokens and repetition penalty, and displays generated Python code with syntax highlighting.

---

## Evaluation

Open and run `Evaluation_results/evaluation.ipynb`. It compares the base model and V1 on 10 Python function prompts and records:

- Generated output, runtime, tokens/second
- Syntax validity (raw and after markdown extraction)
- Docstring, return statement, type hint, and edge case signals
- Markdown/prose leakage from the base model

Results are saved to `Evaluation_results/evaluation_results.json`.

### Key Finding

V1 learned the output contract — it generates raw Python code directly with no chat wrapper, no markdown fences, and no explanations. The base model behaves like a tutorial chatbot. V1 is cleaner and more immediately usable as raw code.

---

## Hardware Notes

Designed for limited VRAM (tested on RTX 4050 6GB). The evaluation notebook loads one model at a time and uses:

```python
dtype=torch.float16
device_map="auto"
max_memory={0: "4GiB", "cpu": "16GiB"}
low_cpu_mem_usage=True
```

---

## Known Limitations

- Evaluation set is small (10 prompts) and not a complete benchmark
- Generated code is not executed against unit tests
- Fine-tuned on only 8% of the dataset (300 steps)
- Some outputs require import cleanup before running

---

## Recommended Next Steps

- Add executable unit tests for each evaluation prompt
- Score functional correctness, not just syntax
- Balance the dataset with simple clean examples
- Train for more steps with a balanced dataset
- Track evaluation results across training runs

---

## Stack

`PyTorch` · `Transformers` · `PEFT` · `TRL` · `BitsAndBytes` · `Streamlit` · `HuggingFace`
