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
import finetune_config_full


# Set env variable
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"


# Set this to True to use minimal resources for testing
USE_TEST_CONFIG = False

if USE_TEST_CONFIG:
    print("⚠️  USING TEST CONFIGURATION")
    finetune_config = finetune_config_full.test_config
else: 
    finetune_config = finetune_config_full.finetune_config

# =============================================================================
# INSTRUCTION FORMATTING FUNCTIONS
# =============================================================================

def format_instruction_from_processed_data(examples):
    """
    Format examples from the processed data pipeline.
    Expected input format: {"input": str, "output": str}
    The input already contains the full formatted prompt from new_data_proc.py
    """

    SIMPLE_CONCATENATION = False  # Set to True to simply concatenate input and output

    texts = []
    for i in range(len(examples["input"])):
        input_text = examples["input"][i]
        output_text = examples["output"][i]
        
        if SIMPLE_CONCATENATION:
            # Combine input and output into a single training text
            # It's REALLY written like that in the doc, although it feels strange for me
            # to simply concatenate this stuff
            text = input_text + "\n"+ output_text
            texts.append(text)
        else:
            # Use the full prompt format with instruction and response
            text = f'''<|begin_of_text|><|user|>
            {input_text}<|end_of_turn|>
            <|assistant|>
            {output_text}<|end_of_turn|>'''

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
        model_path = finetune_config["model_id"]
        print(f"Loading base model: {model_path}")
    

    
    # Load model
    model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=finetune_config["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
        device_map="cuda:0",
    )

    # If using base model without LoRA, we need to set it up
    if not finetune_config["use_pretrained_checkpoint"]:
        model = FastLanguageModel.get_peft_model(
        model,
        r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = True,   # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    
    
    # Load custom tokenizer
    print("Loading custom Amharic tokenizer from:", finetune_config["tokenizer_path"])
    tokenizer = AutoTokenizer.from_pretrained(finetune_config["tokenizer_path"])

    # Handle vocabulary size mismatch
    if len(tokenizer) != len(base_tokenizer):
        print(f"Resizing embeddings: {len(base_tokenizer)} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    
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
        # EXPECTED BEHAVIOR: Load from the processed data files. 
        print("Loading from processed Amharic instruction data...")
        
        # Try to load from test_output_amharic directory first
        train_file = "processed_amharic_data/train.json"
        if os.path.exists(train_file):
            print(f"Loading training data from: {train_file}")
            import json
            with open(train_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dataset = Dataset.from_list(data)  
        else:
            # Fallback to creating sample data if files don't exist
            print("Processed data files not found")
            return
            
    
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
    
    # Define which modules to freeze
    # NOTE: We can only freeze LoRA modules from the pretrained model, can't add another layer of LoRA on top!
    modules_to_freeze = []
    
    for name, param in model.named_parameters():
        # Target LoRA parameters, which have 'lora_A' or 'lora_B' in their name
        if 'lora' in name:
            # Check if the parameter belongs to a module we want to freeze
            if any(module_to_freeze in name for module_to_freeze in modules_to_freeze):
                param.requires_grad = False
                print(f"  - Froze parameter: {name}")
    
    # Load and format dataset
    dataset = load_instruction_dataset()
    
    # Format dataset using the processed data format from new_data_proc.py
    formatted_dataset = format_dataset(dataset, tokenizer)
    
    # Don't do eval for now. takes too much time
    if False and len(formatted_dataset) > 100:  # Only split if we have enough data
        train_test_split = formatted_dataset.train_test_split(test_size=0.1, seed=finetune_config["seed"])
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
    else:
        train_dataset = formatted_dataset
        eval_dataset = None
    
    print(f"Training on {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Evaluating on {len(eval_dataset)} examples")
    
    from datetime import datetime
    # Create output directory
    output_dir = os.path.join(
        finetune_config["output_dir"], 
        f"{finetune_config['run_name']}_lr{finetune_config['learning_rate']}",
        datetime.now().strftime("%Y%m%d-%H%M%S")
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
            lr_scheduler_type="linear",
            logging_steps=finetune_config["logging_steps"],
            eval_strategy=finetune_config["eval_strategy"] if eval_dataset else "no",
            eval_steps=finetune_config["eval_steps"] if eval_dataset else None,
            save_steps=finetune_config["save_steps"],
            output_dir=output_dir,
            run_name=finetune_config["run_name"],
            seed=finetune_config["seed"],
            remove_unused_columns=False,
        ),
    )
    
    # Start training
    print("Starting instruction finetuning...")
    trainer_stats = trainer.train()

    # Print model embedding dimensions
    embedding = model.get_input_embeddings()
    print(f"Model embedding dimensions: {embedding.weight.shape}")
    
    # Save final model
    print(f"Saving finetuned model to {output_dir}")
    from peft import PeftModel

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    
    # Save training stats
    import json
    with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
        json.dump(trainer_stats.metrics, f, indent=2)
    
    print("Instruction finetuning completed successfully!")
    print(f"Model saved to: {output_dir}")

    print("Loading the model and tokenizer again to check if everything is fine...")
    try:
        reloaded_model = FastLanguageModel.from_pretrained(
            model_name=output_dir,
            max_seq_length=finetune_config["max_seq_length"],
            dtype=None,
            load_in_4bit=True,  
            device_map="cuda:0",
        )[0]
        reloaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
        print("Reloaded model and tokenizer successfully.")

        print("Testing the reloaded model with a sample prompt...")

        prompt = "ፕላኔቷን ምድር ግለጽ።"

        text = f'''<|begin_of_text|><|user|>
                    {prompt}<|end_of_turn|>
                    <|assistant|>\n'''

        inputs = tokenizer([text], return_tensors="pt").to("cuda")

        from transformers import TextStreamer
        text_streamer = TextStreamer(tokenizer)

        _ = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            repetition_penalty=1.2,
            do_sample=True,
            top_k=8,
            top_p=0.8,
            temperature=0.5
        )
    except Exception as e:
        print(f"Error reloading model or tokenizer: {e}")
    
    return model, tokenizer

if __name__ == "__main__":
    main() 