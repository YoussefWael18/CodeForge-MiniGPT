# Mini GPT Coder V1

Mini GPT Coder V1 is a local fine-tuning project for `Qwen/Qwen2.5-Coder-1.5B-Instruct`. The goal is to train a small coding assistant that generates concise Python functions, compare it against the base model, and expose the merged fine-tuned model through a simple Streamlit interface.

## Project Overview

This project includes:

- Data preparation from local CodeSearchNet-style Parquet files.
- QLoRA fine-tuning of Qwen2.5-Coder-1.5B-Instruct.
- Export of a merged standalone model for local inference.
- Evaluation of the base model versus the fine-tuned V1 model.
- A Streamlit GUI for interactive code generation.
- A Word documentation file summarizing the project.

## Repository Structure

```text
.
|-- codesearchnet/
|   `-- pair/
|       |-- train-00000-of-00003.parquet
|       |-- train-00001-of-00003.parquet
|       `-- train-00002-of-00003.parquet
|-- data preprocessing/
|   |-- data_prep.py
|   `-- Processed_dataset/
|       `-- golden_train.jsonl
|-- Evaluation_results/
|   |-- evaluation.ipynb
|   `-- evaluation_results.json
|-- gui/
|   |-- app.py
|   `-- README.md
|-- mini-gpt-coder-merged/
|   |-- model.safetensors
|   |-- config.json
|   |-- tokenizer.json
|   `-- generation_config.json
|-- training.ipynb
|-- project_documentation.docx
`-- README.md
```

## Model

- Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- Fine-tuned model: V1, trained for 300 steps
- Local merged model path: `mini-gpt-coder-merged/`
- Training method: QLoRA / PEFT LoRA
- Main task: Python function generation from short natural-language prompts

## Data Preparation

The data preparation script loads the local Parquet shards from `codesearchnet/pair/`, filters examples by instruction and code length, formats them in a ChatML-style prompt/response structure, and creates a curated JSONL training set.

Run from the project root:

```powershell
python "data preprocessing/data_prep.py"
```

The current processed dataset is stored at:

```text
data preprocessing/Processed_dataset/golden_train.jsonl
```

Note: `training.ipynb` currently references `datasets/golden_train.jsonl`. Before training, make sure the notebook path matches the actual dataset location, or place a copy of `golden_train.jsonl` where the notebook expects it.

## Training

Open and run:

```text
training.ipynb
```

The notebook:

1. Loads the processed JSONL dataset.
2. Loads `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
3. Applies 4-bit quantization for memory-efficient training.
4. Configures LoRA adapters.
5. Runs supervised fine-tuning for 300 steps.
6. Saves LoRA adapters.
7. Merges adapters into a standalone model.
8. Saves the merged model to `mini-gpt-coder-merged/`.

Important training settings:

- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.05`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Batch size: `1`
- Gradient accumulation steps: `4`
- Learning rate: `2e-4`
- Max sequence length: `2048`
- Max steps: `300`

## Evaluation

The evaluation notebook is located at:

```text
Evaluation_results/evaluation.ipynb
```

It now has two layers:

1. Raw model comparison: runs the base model and V1 on the same ten Python function prompts.
2. Evidence-first analysis: reloads `evaluation_results.json`, builds richer comparison tables, visualizes behavior differences, and writes a final interpretation.

The notebook records and analyzes:

- Generated output, runtime, tokens generated, and tokens per second.
- Raw syntax validity and syntax validity after extracting code from Markdown fences.
- Docstring, return statement, type-hint, and edge-case handling signals.
- Markdown/prose leakage from the base model.
- Missing import signals such as `math`, `re`, `reduce`, `List`, and `Union`.
- Unwanted `self` usage when the prompt asks for a standalone function.
- Prompt-level quality deltas showing where V1 improved most.

Saved results are available at:

```text
Evaluation_results/evaluation_results.json
```

Current conclusion: V1 added real value mainly by learning the output contract. It behaves more like a direct Python-function generator, while the base model behaves more like a tutorial chatbot. V1 is cleaner and more immediately usable as raw code, but the next improvement target is functional correctness: unit-test pass rate, missing imports, standalone function behavior, and broader prompt coverage.

## Running the GUI

The Streamlit app is located at `gui/app.py`.

Run from the project root:

```powershell
streamlit run gui/app.py
```

The app loads:

```text
mini-gpt-coder-merged/
```

It provides controls for:

- Max new tokens
- Repetition penalty
- Function request text

The generated result is displayed as Python code.

## Requirements

The project uses Python with these main libraries:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `trl`
- `bitsandbytes`
- `streamlit`
- `safetensors`
- `accelerate`

Install dependencies in your environment before training or running inference. GPU support is recommended, especially for training.

## Hardware Notes

This project is designed around limited VRAM. The evaluation notebook loads one model at a time because a 6 GB GPU cannot hold both the base and fine-tuned model simultaneously.

For evaluation, the notebook uses:

```python
dtype=torch.float16
device_map="auto"
max_memory={0: "4GiB", "cpu": "16GiB"}
low_cpu_mem_usage=True
```

The evaluation notebook intentionally avoids `BitsAndBytesConfig` for the merged local model because it can cause stability issues in this environment.

## Known Limitations

- The evaluation set is small and should not be treated as a complete benchmark.
- Current metrics are heuristic and do not execute generated functions against unit tests.
- Some generated code can be syntactically valid but still require imports or context cleanup.
- The fine-tuned model was trained for only 300 steps, so more training may improve consistency.
- The GUI uses 4-bit loading, while evaluation uses fp16 loading for merged-model stability.

## Recommended Next Steps

- Add executable unit tests for each evaluation prompt.
- Score functional correctness, not just syntax and output shape.
- Compare raw base output, code-extracted base output, and V1 output.
- Add checks for missing imports, unwanted `self`, and prompt mismatch.
- Expand the benchmark beyond ten prompts with easy, medium, and edge-case tasks.
- Track evaluation results across training runs and checkpoints.
- Align dataset paths between `data_prep.py`, `training.ipynb`, and the processed dataset folder.

## Documentation

A fuller Word document is included:

```text
project_documentation.docx
```

It covers the project workflow, training configuration, evaluation interpretation, GUI usage, risks, and suggested improvements.
