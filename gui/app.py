from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # same folder as this script
MODEL_PATH = PROJECT_ROOT / "mini-gpt-coder-merged"


st.set_page_config(
    page_title="Mini GPT Coder",
    page_icon="</>",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading merged model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # must match training dtype
        bnb_4bit_quant_type="nf4",
    )

    max_memory = {0: "4GiB", "cpu": "16GiB"} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
        )
    model.eval()

    return model, tokenizer


def format_prompt(user_prompt):
    return (
        "<|im_start|>user\n"
        "Write a Python function for the following:\n"
        f"{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def clean_output(decoded, prompt):
    output = decoded.replace(prompt, "")
    output = output.split("<|im_end|>")[0]
    output = output.replace("<|im_start|>assistant", "")
    return output.strip()


def generate_code(model, tokenizer, user_prompt, max_new_tokens, repetition_penalty):
    prompt = format_prompt(user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    stop_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_token_id,
        )

    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    return clean_output(decoded, prompt)


st.title("Mini GPT Coder")
st.caption("Local Streamlit interface for testing the merged Qwen LoRA model.")

if not MODEL_PATH.exists():
    st.error(f"Model folder not found: {MODEL_PATH}")
    st.stop()

with st.sidebar:
    st.header("Generation")
    max_new_tokens = st.slider("Max new tokens", 64, 512, 256, step=32)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.15, step=0.05)
    st.divider()
    st.write("Model path")
    st.code(str(MODEL_PATH), language="text")


model, tokenizer = load_model()

default_prompt = "Write a Python function to read a CSV file and return the headers."
user_prompt = st.text_area(
    "Function request",
    value=default_prompt,
    height=120,
)

generate = st.button("Generate code", type="primary", use_container_width=True)

if generate:
    if not user_prompt.strip():
        st.warning("Enter a function request first.")
        st.stop()

    with st.spinner("Generating..."):
        code = generate_code(
            model=model,
            tokenizer=tokenizer,
            user_prompt=user_prompt.strip(),
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )

    st.subheader("Generated Python")
    st.code(code, language="python")