config = {
    # =============================================================================
    # DATASET SETTINGS
    # =============================================================================
    "data_id": "iocuydi/amharic-redpajama-synthetic",
    "fraction": 0.05,
    "data_dir": "data",
    "data_name_base": "amharic-redpajama-synthetic",
    "data_cache_dir": "./cache",
    
    # =============================================================================
    # MODEL AND TOKENIZER SETTINGS
    # =============================================================================
    "tokenizer_path": "amharic_tokenizer/llama3_tokenizer_amharic_extended",
    "model_id": "unsloth/Llama-3.2-1B-bnb-4bit",
    "max_seq_length": 2048,
    
    # =============================================================================
    # LORA CONFIGURATION
    # =============================================================================
    "lora_r": 128,  # LoRA rank - higher = more capacity, Suggested 8, 16, 32, 64, 128, chose higher due to CPT
    "lora_alpha": 32, # LoRA alpha scaling, use 32 for increased stability
    "lora_dropout": 0.0,  # Dropout, 0 for maximal throughput with CPT
    "target_modules": [  # Modules to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head"
    ],
    
    # =============================================================================
    # TRAINING HYPERPARAMETERS
    # =============================================================================

    "full_epoch": True, # If True, runs for 'num_train_epochs' otherwise runs for 'max_steps'
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "max_steps": 100, # useful for quick testing if things are running smoothly
    "num_train_epochs": 1, # if computationally feasible increase to 2-3 to improve learning
    "warmup_steps": 50,
    "learning_rate": 5e-5,
    "embedding_learning_rate": 1e-5,
    "weight_decay": 0.01,
    
    # =============================================================================
    # OUTPUT SETTINGS
    # =============================================================================
    "lora_cpt_output_prefix": "lora_cpt_",
    "check_output_dir": None,          # Where to save checkpoints (move to scratch)
    
    # =============================================================================
    # SYSTEM SETTINGS
    # =============================================================================
    "seed": 42,
}