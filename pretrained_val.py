from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch
import math
from datasets import load_from_disk


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="trainer_output/checkpoint-44000",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda:0",
)

from datasets import load_dataset

dataset = load_from_disk("data/amharic-redpajama-synthetic_001")

# Get bottom 0.1% of the dataset
num_samples = max(1, int(len(dataset) * 0.001))
bottom_dataset = dataset.select(range(len(dataset) - num_samples, len(dataset)))


FastLanguageModel.for_inference(model)

# Step 3: Tokenize and compute loss
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

# Step 4: Calculate perplexity
ppl = compute_perplexity(model, tokenizer, bottom_dataset)
print(f"Perplexity on bottom 0.1% of dataset: {ppl}")