#!/usr/bin/env python3
"""
Minimal test script for Amharic instruction finetuning - CPU only, very limited config.
"""
from instruction_finetuning import main
import torch
import transformers
import datasets
import unsloth
import sys

def run_minimal_training_test():
    """Run absolute minimal training test."""
    try:
        print("🧪 Running minimal training test...")
        
        # Import and run
        from instruction_finetuning import main
        result = main()
        
        if result is None:
            print("❌ main() returned None - check for errors in the training script")
            return False
            
        model, tokenizer = result
        print("✅ Training test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Minimal test main function."""
    print("🧪 MINIMAL FINETUNING TEST")
    
    if run_minimal_training_test():
        print("🎉 SUCCESS - Training works!")
    else:
        print("❌ FAILED - Check errors above")

if __name__ == "__main__":
    main() 