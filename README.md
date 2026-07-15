# Instruction Fine-Tuning of a Language Model with LoRA — AI in Healthcare

An end-to-end, parameter-efficient fine-tuning (PEFT) pipeline that adapts a
pre-trained causal language model (**GPT-2, 124M**) to the *AI-in-healthcare*
domain using **LoRA** via Hugging Face `peft` and TRL's `SFTTrainer`. The
project covers the full workflow: synthetic instruction-dataset construction,
Alpaca-style prompt formatting, LoRA adapter training, and a structured
before/after evaluation of the adapted model.

> **A note on naming.** This project implements **LoRA**, not full **QLoRA**.
> QLoRA's defining feature is 4-bit NF4 quantization via `bitsandbytes`, which
> requires a CUDA GPU. Because the experiment was run on **CPU**, quantization
> is disabled and the base model is loaded in full `float32` precision. The
> code is written so that switching to QLoRA on a GPU is a small, documented
> change (see the commented `BitsAndBytesConfig` block in
> [`src/train_lora.py`](src/train_lora.py)). Naming this honestly is
> deliberate — the LoRA method itself is what's demonstrated here.

---

## Why this project

Large models are expensive to fine-tune fully. LoRA freezes the pre-trained
weights and injects small low-rank trainable matrices into the attention
layers, so only a tiny fraction of parameters is updated. This project shows,
concretely, that this works: a frozen GPT-2 is adapted to a new domain by
training **~0.24% of its parameters**, and the behavioural shift is visible in
a direct before/after comparison.

## Method at a glance

| Component | Choice |
|---|---|
| Base model | GPT-2 (124M), causal LM |
| Technique | LoRA (PEFT) — base weights frozen |
| Target module | `c_attn` (GPT-2's fused Q/K/V projection) |
| LoRA config | rank `r=8`, `alpha=16`, dropout `0.05` |
| Trainable params | **294,912 (~0.24%)** of 124M |
| Dataset | ~510 instruction–response pairs, JSONL / Alpaca format |
| Split | 90% train / 10% validation (seed 42) |
| Optimizer | AdamW, lr `2e-4`, cosine schedule, warmup `0.03` |
| Batch | size 2 × grad-accum 4 = effective 8 |
| Epochs | 5 |
| Trainer | Hugging Face TRL `SFTTrainer` |
| Runtime | CPU, ~22 minutes |

## Dataset

The dataset (`data/ai_healthcare_instructions.jsonl`) is a synthetic set of
instruction–response pairs generated from the review article *"Artificial
Intelligence in Healthcare: Applications, Challenges, and Future Directions"*
(2025), using an LLM to convert source material into a conversational Q&A
format. Each line is a JSON object:

```json
{"instruction": "How does AI assist in surgical procedures?",
 "response": "AI provides real-time guidance and enhances precision during surgical navigation, reducing risks and improving outcomes."}
```

Topics span diagnostics, radiology and medical imaging, pathology, surgical
assistance, drug discovery, genomics, personalized medicine, and the ethics of
clinical AI.

## Results

Training and validation loss both decrease steadily across the five epochs,
with validation loss falling sharply early and then stabilizing — and no
upward drift, i.e. no sign of overfitting.

![Loss curve](results/loss_curve.png)

| Epoch | Train loss | Val loss |
|:---:|:---:|:---:|
| 1 | 2.17 | 1.63 |
| 5 | 0.26 | 0.27 |

**Before vs. after fine-tuning.** On a fixed set of 10 held-out healthcare
prompts, the base model frequently hallucinated — inventing fictional academic
identities, fabricated studies, and off-topic first-person narratives. After
LoRA fine-tuning, the adapted model consistently:

- stayed within the healthcare domain and addressed the correct subject
  (drug discovery, ICU prediction, robotic surgery, genomics, mental health);
- followed the instruction / response structure it was trained on;
- used domain vocabulary (diagnostic accuracy, clinical decision-making,
  patient outcomes) absent from the base model.

Example:

> **Prompt:** *How is AI integrated into robotic surgical systems to reduce operative risk?*
>
> **Before (base GPT-2):** fabricated a research study and made unsupported
> claims about an "automated robotic surgical system."
>
> **After (GPT-2 + LoRA):** *"AI can be applied in robotic systems to reduce
> operative risk by reducing the amount of effort required to complete an
> operation."* — concise, on-topic, and correctly scoped.

## Honest limitations

This is a small-scale demonstration, and the write-up treats its limitations
as first-class results rather than hiding them:

1. **Repetition / low specificity.** GPT-2 at 124M has limited capacity to
   store new facts; LoRA reliably transfers *style and domain vocabulary* more
   than specific factual knowledge, so some responses are generic or repetitive.
2. **Small base model.** A larger base (e.g. Mistral 7B, Llama-2 7B) would give
   substantially more specific, informative outputs from the same pipeline.
3. **CPU / no quantization.** Running on CPU precluded 4-bit NF4 quantization,
   so this is LoRA rather than full QLoRA.
4. **Short training on synthetic data.** More epochs on a richer,
   higher-specificity dataset would reduce repetition and improve diversity.

The takeaway: the **LoRA technique itself is sound and efficient**; the ceiling
here is set by hardware and data scale, not by the method.

## Repository structure

```
healthcare-llm-lora/
├── README.md
├── requirements.txt
├── data/
│   └── ai_healthcare_instructions.jsonl   # ~510 instruction-response pairs
├── src/
│   ├── train_lora.py                       # LoRA fine-tuning pipeline
│   ├── evaluate.py                         # before/after inference comparison
│   └── plot_loss.py                        # loss-curve visualization
├── results/
│   ├── training_metrics.json               # per-epoch train/val loss
│   └── loss_curve.png
└── report/
    └── written_report.md                   # full technical write-up
```

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train the LoRA adapter (writes results/trained_adapter + metrics)
python src/train_lora.py --data data/ai_healthcare_instructions.jsonl --out results/

# 3. Plot the loss curve
python src/plot_loss.py

# 4. Compare base vs. fine-tuned model on the test prompts
python src/evaluate.py --adapter results/trained_adapter
```

To run **true QLoRA** on a CUDA GPU, uncomment the `BitsAndBytesConfig` block
in `src/train_lora.py` and set a larger `BASE_MODEL_ID`.

## Skills demonstrated

Parameter-efficient fine-tuning (LoRA/PEFT) · Hugging Face `transformers`,
`peft`, `trl`, `datasets` · instruction tuning & Alpaca prompt formatting ·
synthetic dataset construction · supervised fine-tuning with `SFTTrainer` ·
train/validation methodology and loss analysis · qualitative LLM evaluation ·
honest experimental reporting.

## License

MIT — see `LICENSE`.
