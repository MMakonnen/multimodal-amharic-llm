from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
    UnslothTrainer,
    UnslothTrainingArguments,
)
import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


# Load config
from config import config

# Set env variable
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

# =============================================================================
# TEST CONFIGURATION (for limited hardware)
# =============================================================================

# Set this to True to use minimal resources for testing
USE_TEST_CONFIG = True

if USE_TEST_CONFIG:
    print("⚠️  USING TEST CONFIGURATION")
    test_config = {
        # Minimal data settings
        "instruction_dataset_id": None,
        "instruction_dataset_path": None,
        "max_seq_length": 256,  # Very small
        "dataset_text_field": "text",
        
        # Model settings
        "base_model_path": None,
        "use_pretrained_checkpoint": False,
        
        # Minimal training settings
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,  # No accumulation
        "num_train_epochs": 1,
        "max_steps": 2,  # Just 2 steps
        "warmup_steps": 0,  # No warmup
        "learning_rate": 1e-5,  # Very low LR
        "embedding_learning_rate": 1e-6,
        "weight_decay": 0.0,
        
        # Minimal LoRA settings
        "lora_r": 4,  # Very low rank
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj"],  # Just one module
        
        # Output settings
        "output_dir": "test_finetuned_models",
        "run_name": "minimal_test",
        
        # Minimal logging
        "eval_strategy": "no",
        "eval_steps": None,
        "save_steps": 999,  # Don't save during test
        "logging_steps": 1,
        
        # System settings
        "seed": 42,
    }
    # Override the main config
    finetune_config = test_config

# =============================================================================
# FINETUNING CONFIGURATION
# =============================================================================
else: 
    finetune_config = {
        # Data settings for instruction finetuning
        "instruction_dataset_id": None,  # Set to your instruction dataset ID
        "instruction_dataset_path": None,  # Or path to local dataset (e.g., "test_output_amharic")
        "max_seq_length": 2048,
        "dataset_text_field": "text",
        
        # Model settings - can use continued pretrained model
        "base_model_path": None,  # Path to your continued pretrained model, or use config["model_id"]
        "use_pretrained_checkpoint": True,  # Whether to load from continued pretraining checkpoint
        
        # Training hyperparameters for instruction finetuning
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 3,
        "max_steps": -1,  # -1 to use epochs instead
        "warmup_steps": 100,
        "learning_rate": 2e-4,  # Higher LR for instruction finetuning  
        "embedding_learning_rate": 5e-5,
        "weight_decay": 0.01,
        
        # LoRA settings for finetuning (can be different from pretraining)
        "lora_r": 64,  
        "lora_alpha": 16,
        "lora_dropout": 0.1,  
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # I'm not sure about concrete modules that we should train with LoRA
        
        # Output settings
        "output_dir": "finetuned_models",
        "run_name": "amharic_instruction_finetune",
        
        # Evaluation settings
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_steps": 200,
        "logging_steps": 10,
        
        # System settings
        "seed": 42,
    }

# =============================================================================
# INSTRUCTION FORMATTING FUNCTIONS
# =============================================================================

def format_instruction_from_processed_data(examples):
    """
    Format examples from the processed data pipeline.
    Expected input format: {"input": str, "output": str}
    The input already contains the full formatted prompt from new_data_proc.py
    """
    texts = []
    for i in range(len(examples["input"])):
        input_text = examples["input"][i]
        output_text = examples["output"][i]
        
        # Combine input and output into a single training text
        # It's REALLY written like that in the doc, although it feels strange for me
        # to simply concatenate this stuff
        text = input_text + "\n"+ output_text
        texts.append(text)
    
    return {"text": texts}

# =============================================================================
# MODEL AND TOKENIZER LOADING
# =============================================================================

def load_model_and_tokenizer():
    """Load the model and tokenizer for instruction finetuning."""
    
    # Determine which model to load
    if finetune_config["use_pretrained_checkpoint"] and finetune_config["base_model_path"]:
        model_path = finetune_config["base_model_path"]
        print(f"Loading continued pretrained model from: {model_path}")
    else:
        model_path = config["model_id"]
        print(f"Loading base model: {model_path}")
    

    
    # Load model
    model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=finetune_config["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
        device_map="cuda:0",
    )
    
    
    # Load custom tokenizer
    print("Loading custom Amharic tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    
    # Handle vocabulary size mismatch
    if len(tokenizer) != len(base_tokenizer):
        print(f"Resizing embeddings: {len(base_tokenizer)} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    # Set up tokenizer properties
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

# =============================================================================
# DATASET LOADING AND PROCESSING
# =============================================================================

def load_instruction_dataset():
    """Load and format instruction dataset."""
    
    if finetune_config["instruction_dataset_path"]:
        # Load from local path
        print(f"Loading dataset from local path: {finetune_config['instruction_dataset_path']}")
        dataset = Dataset.load_from_disk(finetune_config["instruction_dataset_path"])
    elif finetune_config["instruction_dataset_id"]:
        # Load from HuggingFace Hub
        print(f"Loading dataset from Hub: {finetune_config['instruction_dataset_id']}")
        dataset = load_dataset(finetune_config["instruction_dataset_id"], split="train")
    else:
        # Load from the processed data files
        print("Loading from processed Amharic instruction data...")
        
        # Try to load from test_output_amharic directory first
        train_file = "test_output_amharic/train.json"
        if os.path.exists(train_file):
            print(f"Loading training data from: {train_file}")
            import json
            with open(train_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)
        else:
            # Fallback to creating sample data if files don't exist
            print("Processed data files not found")
            
    
    print(f"Loaded {len(dataset)} instruction examples.")
    return dataset

def format_dataset(dataset, tokenizer):
    """Format the dataset for instruction finetuning using processed data format."""
    
    # Format using the processed data format
    formatted_dataset = dataset.map(format_instruction_from_processed_data, batched=True)
    
    # Add EOS token
    def add_eos_token(examples):
        return {"text": [text + tokenizer.eos_token for text in examples["text"]]}
    
    formatted_dataset = formatted_dataset.map(add_eos_token, batched=True)
    
    return formatted_dataset

# =============================================================================
# MAIN FINETUNING FUNCTION
# =============================================================================

def main():
    """Main finetuning function."""
    
    print("=" * 60)
    print("AMHARIC LLM INSTRUCTION FINETUNING")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Set up LoRA
    print("Setting up LoRA adapters for instruction finetuning...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=finetune_config["lora_r"],
        target_modules=finetune_config["target_modules"],
        lora_alpha=finetune_config["lora_alpha"],
        lora_dropout=finetune_config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=finetune_config["seed"],
        use_rslora=True,
        loftq_config=None,
    )
    
    # Load and format dataset
    dataset = load_instruction_dataset()
    
    # Format dataset using the processed data format from new_data_proc.py
    formatted_dataset = format_dataset(dataset, tokenizer)
    
    # Split dataset if needed
    if len(formatted_dataset) > 100:  # Only split if we have enough data
        train_test_split = formatted_dataset.train_test_split(test_size=0.1, seed=finetune_config["seed"])
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
    else:
        train_dataset = formatted_dataset
        eval_dataset = None
    
    print(f"Training on {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Evaluating on {len(eval_dataset)} examples")
    
    # Create output directory
    output_dir = os.path.join(
        finetune_config["output_dir"], 
        f"{finetune_config['run_name']}_r{finetune_config['lora_r']}_lr{finetune_config['learning_rate']}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up trainer
    print("Setting up trainer...")
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=finetune_config["max_seq_length"],
        dataset_num_proc=2,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=finetune_config["per_device_train_batch_size"],
            gradient_accumulation_steps=finetune_config["gradient_accumulation_steps"],
            num_train_epochs=finetune_config["num_train_epochs"],
            max_steps=finetune_config["max_steps"],
            warmup_steps=finetune_config["warmup_steps"],
            learning_rate=finetune_config["learning_rate"],
            embedding_learning_rate=finetune_config["embedding_learning_rate"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=finetune_config["weight_decay"],
            lr_scheduler_type="cosine",
            logging_steps=finetune_config["logging_steps"],
            eval_strategy=finetune_config["eval_strategy"] if eval_dataset else "no",
            eval_steps=finetune_config["eval_steps"] if eval_dataset else None,
            save_steps=finetune_config["save_steps"],
            output_dir=output_dir,
            report_to=["tensorboard"],  # Enable tensorboard logging
            run_name=finetune_config["run_name"],
            seed=finetune_config["seed"],
            remove_unused_columns=False,
        ),
    )
    
    # Start training
    print("Starting instruction finetuning...")
    trainer_stats = trainer.train()
    
    # Save final model
    print(f"Saving finetuned model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training stats
    import json
    with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
        json.dump(trainer_stats.metrics, f, indent=2)
    
    print("Instruction finetuning completed successfully!")
    print(f"Model saved to: {output_dir}")
    
    return model, tokenizer

if __name__ == "__main__":
    main() 