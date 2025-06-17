from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
    UnslothTrainer,
    UnslothTrainingArguments,
)
import os
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer

# Load config
from config import config

# Set env variable
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # Disable CCE since it's not supported for CPT
# -> CHECK IF EQUIVALENT TO: %env UNSLOTH_RETURN_LOGITS=1 # Run this to disable CCE since it is not supported for CPT

# =============================================================================
# MODEL LOADING
# =============================================================================

print("Loading base model...")
model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["model_id"],
    max_seq_length=config["max_seq_length"],
    dtype=None,
    load_in_4bit=True,
)

print("Loading custom Amharic tokenizer...")
custom_tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])

# Resize token embeddings if vocab size changed
original_vocab_size = len(base_tokenizer)
new_vocab_size = len(custom_tokenizer)
if new_vocab_size != original_vocab_size:
    print(f"Resizing embeddings: {original_vocab_size} → {new_vocab_size}")
    model.resize_token_embeddings(new_vocab_size)
tokenizer = custom_tokenizer

# Ensure EOS token is defined
if tokenizer.eos_token is None:
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")

# =============================================================================
# LORA SETUP
# =============================================================================
print("Setting up LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora_r"],
    target_modules=config["target_modules"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=config["seed"],
    use_rslora=True,
    loftq_config=None,
)

# =============================================================================
# DATA LOADING
# =============================================================================

processed_datasets = []

for dataset_config in config["datasets"]:

    percent_int = int(dataset_config["fraction"] * 100)
    fraction_suffix = f"_{percent_int:03d}"
    dataset_path = os.path.join(dataset_config["data_dir"], dataset_config["data_name_base"] + fraction_suffix)
    # -> CHECK IF LAST LINE IS CORRECT DATA PATH, should be like: "data/amharic-redpajama-synthetic_005"

    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"Loaded {len(dataset)} samples.")

    # Format text with EOS
    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        return {"text": [text + EOS_TOKEN for text in examples["text"]]}

    print("Formatting dataset...")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    processed_datasets.append(dataset)

# Concatenate all datasets
dataset = concatenate_datasets(processed_datasets)

# -> DOUBLE CHECK IF FORMAT CORRECTLY FOR MODEL !!!

# =============================================================================
# TRAINER SETUP
# =============================================================================

# Create dynamic output directory under pretrained_models/
model_id_clean = config["model_id"].split("/")[-1]
seed = config.get("seed", 42)

output_name = f"{config['lora_cpt_output_prefix']}{model_id_clean}-seed{seed}"
output_dir = os.path.join("pretrained_models", output_name)
os.makedirs(output_dir, exist_ok=True)


print("Setting up trainer...")
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=config["max_seq_length"],
    dataset_num_proc=4,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs = config['num_train_epochs'] if config.get('full_epoch', False) else -1,
        max_steps = -1 if config.get('full_epoch', False) else config['max_steps'],
        warmup_steps=config["warmup_steps"],
        learning_rate=config["learning_rate"],
        embedding_learning_rate=config["embedding_learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=config["weight_decay"],
        lr_scheduler_type="linear",
        logging_steps=1,
        output_dir=config["check_output_dir"],
        report_to=[],
        seed=config["seed"],
    ),
)

# =============================================================================
# TRAINING
# =============================================================================

print("Starting continued pretraining...")
trainer_stats = trainer.train()
# MIGHT THROW THIS ERROR, IF YES CHECK HERE: https://github.com/unslothai/unsloth/issues/2127

# =============================================================================
# AFTER TRAINING
# =============================================================================

# Save final model
print(f"Saving model to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Continued pretraining completed successfully!")