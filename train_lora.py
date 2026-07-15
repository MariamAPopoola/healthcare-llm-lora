"""
train_lora.py
-------------
LoRA (parameter-efficient) fine-tuning of a causal language model (GPT-2)
on a synthetic AI-in-Healthcare instruction dataset, using Hugging Face
PEFT + TRL's SFTTrainer.

Note on naming: this pipeline implements LoRA, not QLoRA. True QLoRA requires
4-bit NF4 quantization via bitsandbytes, which needs a CUDA GPU. This project
was run on CPU, so quantization is disabled and the base model is loaded in
full float32 precision. The code is structured so that enabling QLoRA on a GPU
is a small change (see the commented BitsAndBytesConfig block below).

Run:
    python src/train_lora.py \
        --data data/ai_healthcare_instructions.jsonl \
        --out results/
"""

import os
import json
import argparse

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_MODEL_ID     = "gpt2"          # GPT-2 (124M) — lightweight, CPU-friendly
LORA_R            = 8               # LoRA rank: adapter capacity
LORA_ALPHA        = 16              # LoRA scaling factor
LORA_DROPOUT      = 0.05
LORA_BIAS         = "none"
TARGET_MODULES    = ["c_attn"]      # GPT-2's fused Q/K/V attention projection
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 5
BATCH_SIZE        = 2
GRAD_ACCUM_STEPS  = 4               # effective batch size = 8
MAX_SEQ_LENGTH    = 256
WARMUP_RATIO      = 0.03
LR_SCHEDULER      = "cosine"
OPTIMIZER         = "adamw_torch"   # paged_adamw_8bit requires a GPU
WEIGHT_DECAY      = 0.001
SEED              = 42


def format_prompt(row: dict) -> dict:
    """Format an (instruction, response) pair into the Alpaca template."""
    return {
        "text": (
            f"### Instruction:\n{row['instruction'].strip()}\n\n"
            f"### Response:\n{row['response'].strip()}<|endoftext|>"
        )
    }


def load_dataset(path: str):
    """Read a JSONL file with 'instruction' and 'response' fields."""
    df = pd.read_json(path, lines=True)
    assert {"instruction", "response"}.issubset(df.columns), \
        "Dataset must contain 'instruction' and 'response' columns."
    df = df[["instruction", "response"]].dropna().reset_index(drop=True)
    print(f"Loaded {len(df)} instruction-response pairs.")

    hf_dataset = Dataset.from_pandas(df).map(format_prompt)
    split = hf_dataset.train_test_split(test_size=0.1, seed=SEED)
    print(f"Train: {len(split['train'])}  |  Validation: {len(split['test'])}")
    return split["train"], split["test"]


def build_model():
    """Load GPT-2 in float32 on CPU and attach a LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token
    tokenizer.padding_side = "right"

    # --- CPU path: full-precision, no quantization -----------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
    )
    # --- GPU / QLoRA path (uncomment on a CUDA machine) ------------------
    # from transformers import BitsAndBytesConfig
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    # from peft import prepare_model_for_kbit_training
    # base_model = prepare_model_for_kbit_training(base_model)

    base_model.config.use_cache = False   # needed for gradient checkpointing

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
        inference_mode=False,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ai_healthcare_instructions.jsonl")
    parser.add_argument("--out", default="results")
    parser.add_argument("--adapter", default="trained_adapter")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(SEED)

    train_dataset, eval_dataset = load_dataset(args.data)
    model, tokenizer = build_model()

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        optim=OPTIMIZER,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        fp16=False,                 # must be False on CPU
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none",
        use_cpu=True,
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    trainer.train()

    # Save the adapter (adapter_model.safetensors + adapter_config.json)
    adapter_path = os.path.join(args.out, args.adapter)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to: {adapter_path}/")

    # Persist per-epoch losses for plotting / reproducibility
    train_log, eval_log = {}, {}
    for entry in trainer.state.log_history:
        ep = entry.get("epoch")
        if ep is None:
            continue
        ep_int = int(round(ep))
        if "loss" in entry:
            train_log[ep_int] = entry["loss"]
        if "eval_loss" in entry:
            eval_log[ep_int] = entry["eval_loss"]

    metrics = {
        "epochs": list(range(1, NUM_EPOCHS + 1)),
        "train_loss": [train_log.get(e) for e in range(1, NUM_EPOCHS + 1)],
        "val_loss": [eval_log.get(e) for e in range(1, NUM_EPOCHS + 1)],
    }
    with open(os.path.join(args.out, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved training_metrics.json")


if __name__ == "__main__":
    main()
