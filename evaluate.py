"""
evaluate.py
-----------
Before / after qualitative evaluation. Runs the same set of test prompts
through (a) the frozen base GPT-2 model and (b) the LoRA-adapted model,
so the effect of fine-tuning can be inspected side by side.

Run (after training):
    python src/evaluate.py --adapter results/trained_adapter
"""

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID  = "gpt2"
MAX_NEW_TOKENS = 128
TEMPERATURE    = 0.7
TOP_P          = 0.9

TEST_PROMPTS = [
    "How are multimodal AI systems used to improve early cancer detection in 2025?",
    "Explain how generative AI accelerates drug discovery and clinical trial design.",
    "What role does AI play in predicting patient deterioration in ICU settings?",
    "Describe the latest advances in AI-powered radiology and medical image analysis.",
    "How does AI enable personalized treatment plans based on genomic data?",
    "What are the applications of large language models in clinical documentation and EHR?",
    "How is AI integrated into robotic surgical systems to reduce operative risk?",
    "Explain AI-powered remote patient monitoring and its impact on chronic disease management.",
    "What are the key ethical challenges of deploying AI diagnostic tools in underserved communities?",
    "How are AI-driven conversational agents being used to support mental health care in 2025?",
]


def run_inference(model, tokenizer, prompts, label):
    """Generate a response for each prompt using the Alpaca template."""
    model.eval()
    responses = []
    print(f"\n{label}\n{'-' * 60}")
    for i, instruction in enumerate(prompts, 1):
        prompt = f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = text.split("### Response:")[-1].strip()
        responses.append(answer)
        print(f"\n[{i}] {instruction}\n    {answer}")
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="results/trained_adapter")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # BEFORE: frozen base model
    before_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float32, device_map={"": "cpu"})
    before = run_inference(before_model, tokenizer, TEST_PROMPTS,
                           "BEFORE FINE-TUNING (base GPT-2)")

    # AFTER: base model + LoRA adapter
    after_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float32, device_map={"": "cpu"})
    after_model = PeftModel.from_pretrained(after_model, args.adapter)
    after = run_inference(after_model, tokenizer, TEST_PROMPTS,
                          "AFTER FINE-TUNING (GPT-2 + LoRA)")

    # Side-by-side summary
    print("\n\nSUMMARY\n" + "=" * 60)
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\nPrompt {i + 1}: {prompt}")
        print(f"  BEFORE: {before[i][:160]}")
        print(f"  AFTER : {after[i][:160]}")


if __name__ == "__main__":
    main()
