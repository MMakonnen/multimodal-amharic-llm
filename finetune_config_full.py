finetune_config = {
        # Data settings for instruction finetuning
        "instruction_dataset_id": None,  # Set to your instruction dataset ID
        "instruction_dataset_path": None,  # Or path to local dataset (e.g., "test_output_amharic")
        "max_seq_length": 2048,
        "dataset_text_field": "text",
        
        # Model settings - can use continued pretrained model
        "base_model_path": "trainer_output/checkpoint-59000", # Path to your continued pretrained model, or use config["model_id"]
        "use_pretrained_checkpoint": True,  # False to train from llama3 base, True to train from continued pretrained checkpoint
        "model_id": "rasyosef/Llama-3.2-1B-Amharic",  # Base model ID 
        "tokenizer_path": "./rasyosef_tokenizer_w_special_tokens",  

        # Training hyperparameters for instruction finetuning
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": 4,
        "max_steps": -1,  # -1 to use epochs instead
        "warmup_steps": 100,
        "learning_rate": 5e-5, 
        "embedding_learning_rate": 1e-5,
        "weight_decay": 0.00,
        
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

test_config = {
        # Minimal data settings
        "instruction_dataset_id": None,
        "instruction_dataset_path": None,
        "max_seq_length": 256,  # Very small
        "dataset_text_field": "text",
        
        # Model settings
        "base_model_path": "pretrained_models/lora_cpt_Llama-3.2-1B-bnb-4bit-seed42",
        "use_pretrained_checkpoint": False,
        
        "model_id": "rasyosef/Llama-3.2-1B-Amharic",  # Base model ID 
        "tokenizer_path": "./rasyosef_tokenizer_w_special_tokens",  

        # Minimal training settings
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,  # No accumulation
        "num_train_epochs": 1,
        "max_steps": 2,  # Just 2 steps
        "warmup_steps": 0,  # No warmup
        "learning_rate": 1e-5,  # Very low LR
        "embedding_learning_rate": 1e-6,
        "weight_decay": 0.0,
        
        
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