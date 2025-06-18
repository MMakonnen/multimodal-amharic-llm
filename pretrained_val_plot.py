import os
import torch
import math
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# Load dataset
dataset = load_from_disk("data/amharic-redpajama-synthetic_001")
num_samples = max(1, int(len(dataset) * 0.001))
bottom_dataset = dataset.select(range(len(dataset) - num_samples, len(dataset)))

# Function to compute perplexity
def compute_perplexity(model, tokenizer, dataset, max_length=2048):
    total_loss = 0.0
    total_tokens = 0

    for example in dataset:
        input_text = example["text"]
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# Iterate through checkpoints
perplexities = []
checkpoints = range(1000, 24001, 1000)

for step in checkpoints:
    checkpoint_path = f"trainer_output/checkpoint-{step}"
    print(f"Evaluating {checkpoint_path}...")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            device_map="cuda:0",
        )

        FastLanguageModel.for_inference(model)
        ppl = compute_perplexity(model, tokenizer, bottom_dataset)
        print(f"Perplexity at checkpoint {step}: {ppl:.2f}")
        perplexities.append(ppl)
    except Exception as e:
        print(f"Error loading or evaluating checkpoint {step}: {e}")
        perplexities.append(None)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(checkpoints, perplexities, marker='o', linestyle='-', color='blue')
plt.title("Perplexity over Training Steps")
plt.xlabel("Checkpoint Step")
plt.ylabel("Perplexity")
plt.grid(True)
plt.xticks(checkpoints, rotation=45)
plt.tight_layout()
plt.savefig("perplexity_plot.png")
print("Plot saved as perplexity_plot.png")
