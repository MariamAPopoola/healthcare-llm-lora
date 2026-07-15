# Written Report — LoRA Instruction Fine-Tuning on an AI-Healthcare Dataset

## 1. Synthetic data: creation process

The synthetic dataset was generated from the review article *"Artificial
Intelligence in Healthcare: Applications, Challenges, and Future Directions"*
(2025). The goal was to build structured instruction–response pairs reflecting
the key concepts, findings, and discussions in the article.

An LLM (Gemini) was used to transform knowledge from the article into a
conversational format. Relevant themes — the role of AI in healthcare,
diagnostic applications, surgical assistance, and emerging technologies — were
identified and used to construct instructional prompts (questions), each paired
with a concise, contextually relevant answer grounded in the source material.

The dataset is stored in JSONL format; each entry has two fields, `instruction`
and `response`. The generation process covered diverse question types —
definitions, applications, methodologies, and significance — to simulate
realistic interactions and support training/evaluation of NLP models. The
final dataset contains ~510 substantive, domain-specific pairs.

**Sample entries:**

1. *Instruction:* What is the primary goal of artificial intelligence in healthcare research?
   *Response:* The primary goal is to enhance healthcare delivery by analyzing data, improving decision-making, and supporting medical research.
2. *Instruction:* How does AI assist in surgical procedures?
   *Response:* AI provides real-time guidance and enhances precision during surgical navigation, reducing risks and improving outcomes.
3. *Instruction:* How is AI applied in pathology?
   *Response:* Artificial intelligence is used in pathology to improve cancer diagnosis by delivering faster and more accurate results.

## 2. LoRA configuration

> **Naming correction.** This experiment implements **LoRA**, not **QLoRA**.
> QLoRA requires 4-bit NF4 quantization via `bitsandbytes`, which needs a CUDA
> GPU. Because the run was on CPU, quantization was disabled and the model was
> loaded in full `float32`. The pipeline is otherwise identical to a QLoRA
> setup, and switching to QLoRA on a GPU is a small change.

The base model is **GPT-2 (124M)**, an open-source causal language model chosen
for its lightweight architecture and suitability for CPU fine-tuning. The LoRA
adapter used rank `r = 8`, scaling factor `alpha = 16`, and dropout `0.05`. The
target module was `c_attn` — GPT-2's fused query/key/value attention
projection. With these settings, only **294,912 parameters (~0.24% of 124M)**
were trained; all base weights stayed frozen.

## 3. Training process

The adapter was trained for **5 epochs** on **459 training samples**, with
**51 held out for validation** (90/10 split). Training used AdamW
(learning rate `2e-4`), a cosine scheduler, batch size 2, and gradient
accumulation over 4 steps (effective batch size 8). Max sequence length was
256 tokens. Training used TRL's `SFTTrainer` on CPU and completed in ~22
minutes.

Training loss fell steadily from **2.17 (epoch 1) to 0.26 (epoch 5)**;
validation loss followed the same downward trend, from **1.63 to 0.27** —
confirming the model learned from the dataset without significant overfitting.

## 4. Results: before vs. after fine-tuning

Across 10 held-out prompts, the base GPT-2 model produced off-topic,
hallucinatory, instruction-unaware text. After fine-tuning, the adapted model
showed clear domain alignment, better instruction-following, and consistent
healthcare vocabulary.

**Improvements observed**

1. **Reduced hallucination.** The base model fabricated fictional academic
   identities, invented studies, and produced unrelated first-person
   narratives. These patterns largely disappeared after fine-tuning; responses
   stayed in the healthcare domain.
2. **Instruction-following.** The base model ignored the instruction structure
   (e.g. answering a drug-discovery question with a personal biography). The
   fine-tuned model recognized the instruction/response format and addressed
   the correct subject area.
3. **Domain vocabulary.** The fine-tuned model consistently used terms such as
   *medical processes, diagnostic accuracy, clinical decision-making, patient
   outcomes* — absent from the base model.
4. **Response structure.** Outputs converged on the Alpaca instruction/response
   pattern used during training.

**Limitations observed**

1. **Repetition.** Some phrasing recurs across prompts — a consequence of a
   small model and limited-diversity training data.
2. **Low specificity.** The model identifies the right domain but rarely gives
   specific factual detail; GPT-2's 124M capacity limits factual injection via
   LoRA.
3. **Occasional looping.** Small causal LMs fine-tuned on short instruction
   data can emit repeated tokens without robust end-of-sequence handling.
4. **CPU constraint.** No 4-bit quantization was possible on CPU, restricting
   this to LoRA (not QLoRA) and to a small base model.

## 5. Loss curve

Both training and validation loss decrease consistently across epochs.
Validation loss drops sharply early and then stabilizes, with no upward drift —
no signs of overfitting. The model converges around epochs 5–6, suggesting
limited benefit from longer training on this dataset. See
`results/loss_curve.png`.

## 6. Conclusion

**Reflections.** The exercise covered the full instruction-tuning pipeline —
synthetic data preparation, prompt formatting, LoRA configuration, supervised
training, and qualitative evaluation. Watching training loss fall from 2.17 to
0.26 and seeing outputs shift from hallucinated biographies to
healthcare-domain responses made the mechanics of fine-tuning concrete.

**Difficulties.** The main technical difficulty was `bitsandbytes`
incompatibility on the CPU runtime (missing CUDA binary / `triton` errors),
which prevented 4-bit quantization and meant the experiment implemented LoRA
rather than full QLoRA. A second constraint was the scale and specificity of
the synthetic dataset, which limited factual richness in the after-responses.

**Potential.** Despite these constraints, the experiment demonstrates the core
value of LoRA: a frozen pre-trained model can be adapted to a new domain by
training well under 0.25% of its parameters on a small, domain-specific
dataset. Three improvements would make the results markedly stronger:
(1) a larger base model (e.g. Mistral 7B / Llama-2 7B) on a GPU, enabling true
QLoRA; (2) a higher-specificity dataset drawn directly from source findings;
and (3) more training epochs on that richer data to reduce repetition. The LoRA
method itself is sound and well-validated; the limitations here are of hardware
and data scale, not of the technique.
