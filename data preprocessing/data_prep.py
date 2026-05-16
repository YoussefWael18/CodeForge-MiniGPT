from datasets import load_dataset

# STEP 1: LOAD THE RAW DATA
# We start by loading our local Parquet files. Think of Parquet files 
# like highly compressed, super-fast spreadsheets.

raw_dataset = load_dataset(
    "parquet",
    data_files=[
        "codesearchnet/pair/train-00000-of-00003.parquet",
        "codesearchnet/pair/train-00001-of-00003.parquet",
        "codesearchnet/pair/train-00002-of-00003.parquet",
    ],
    split="train",
)

print("Total rows:", len(raw_dataset), flush=True)
print("Column names:", raw_dataset.column_names, flush=True)


# STEP 2: THE "MEMORY SAVER" FILTER
# AI models have a limited "short-term memory" (Context Window). 
# If we feed it giant blocks of code, the graphics card will crash.
# Here, we count the words and only keep the suitable sizes.

def word_count(value):
    # Splits a sentence into a list of words and counts them
    return len((value or "").split())

def keep_batch(batch):
    # Count words for both the human comments and the Python code
    doc_word_counts = [word_count(doc) for doc in batch["comment"]]
    code_word_counts = [word_count(code) for code in batch["code"]]

    # Keep it ONLY IF:
    # 1. The instruction is short and sweet (under 50 words)
    # 2. The code isn't just a 1-line stub (over 15 words)
    # 3. The code isn't a massive script (under 200 words)
    return [
        doc_words <= 50 and 15 <= code_words <= 200
        for doc_words, code_words in zip(doc_word_counts, code_word_counts)
    ]

# Apply the filter across the whole dataset in batches for speed
filtered_dataset = raw_dataset.filter(keep_batch, batched=True)
print("Filtered rows:", len(filtered_dataset), flush=True)


# STEP 3: FORMATTING FOR THE AI
# The AI needs to understand this data as a conversation. 
# We wrap the data in special "ChatML" tags so it knows exactly 
# when the user is asking a question and when the AI is answering.

def format_batch(batch):
    return {
        "text": [
            (
                "<|im_start|>user\n"
                "Write a Python function for the following:\n"
                f"{docstring}<|im_end|>\n"
                "<|im_start|>assistant\n"
                f"{code}<|im_end|>"
            )
            # Combine the docstring and code row by row
            for docstring, code in zip(batch["comment"], batch["code"])
        ]
    }

# Apply the formatting and throw away the old, raw columns
formatted_dataset = filtered_dataset.map(
    format_batch,
    batched=True,
    remove_columns=filtered_dataset.column_names,
)


# STEP 4: SHUFFLE AND SAVE THE "GOLDEN SUBSET"
# We still have way too much data to train in a short amount of time.
# We shuffle the deck to mix up the examples, pull out exactly 15,000, 
# and save it in a lightweight format (.jsonl) ready for the cloud.

golden_subset = formatted_dataset.shuffle(seed=42).select(range(15_000))
golden_subset.to_json("golden_train.jsonl", orient="records", lines=True)

print("Saved golden_train.jsonl successfully.", flush=True)