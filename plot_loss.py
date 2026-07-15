"""
plot_loss.py
------------
Plots training vs. validation loss per epoch from results/training_metrics.json
(written by train_lora.py). Saves results/loss_curve.png.

Run:
    python src/plot_loss.py
"""

import os
import json
import argparse

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/training_metrics.json")
    parser.add_argument("--out", default="results/loss_curve.png")
    args = parser.parse_args()

    with open(args.metrics) as f:
        m = json.load(f)

    epochs = m["epochs"]
    train_loss = m["train_loss"]
    val_loss = m["val_loss"]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", linewidth=2,
             color="#2563eb", label="Training loss")
    plt.plot(epochs, val_loss, marker="s", linewidth=2,
             color="#16a34a", label="Validation loss")

    for x, y in zip(epochs, train_loss):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, color="#2563eb")
    for x, y in zip(epochs, val_loss):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=8, color="#16a34a")

    plt.title("LoRA Fine-Tuning: Training vs. Validation Loss\n"
              "GPT-2 on AI-Healthcare Instruction Dataset", fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
