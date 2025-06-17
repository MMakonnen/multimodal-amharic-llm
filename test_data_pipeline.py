"""
Test script for the new data processing pipeline.
"""

import os
import json
from new_data_proc import process_datasets_pipeline, create_datasets_from_processed_data

def test_small_sample():
    """Test the pipeline with a small sample of data."""
    
    print("=== TESTING NEW PIPELINE ===\n")
    
    # Test with small sample and different configurations
    print("Testing with different configurations...")
    
    # Test 1: Only Amharic examples
    print("\n1. Testing Amharic-only configuration...")
    try:
        train_data, test_data = process_datasets_pipeline(
            with_translation=False,
            allow_english=False,
            with_amharic=True,
            train_ratio=0.8,
            output_dir='test_output_amharic'
        )
        print(f"✅ Success: {len(train_data)} train, {len(test_data)} test examples")
        
        # Show sample
        if train_data:
            print("Sample training example:")
            print(f"Input: {train_data[0]['input'][:100]}...")
            print(f"Output: {train_data[0]['output'][:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: With English and translation
    print("\n2. Testing full configuration...")
    try:
        train_data, test_data = process_datasets_pipeline(
            with_translation=True,
            allow_english=True,
            with_amharic=True,
            train_ratio=0.8,
            output_dir='test_output_full'
        )
        print(f"✅ Success: {len(train_data)} train, {len(test_data)} test examples")
        
        # Show different types of examples
        if len(train_data) > 3:
            print("\nSample examples:")
            for i, example in enumerate(train_data[:3]):
                print(f"\nExample {i+1}:")
                print(f"Input: {example['input'][:100]}...")
                print(f"Output: {example['output'][:100]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Convert to HuggingFace Dataset
    print("\n3. Testing Dataset conversion...")
    try:
        if 'train_data' in locals() and 'test_data' in locals():
            train_dataset, test_dataset = create_datasets_from_processed_data(train_data, test_data)
            print(f"✅ Success: Created HF datasets with {len(train_dataset)} train, {len(test_dataset)} test")
            print(f"Dataset columns: {train_dataset.column_names}")
    except Exception as e:
        print(f"❌ Error: {e}")

def validate_output_files():
    """Validate that output files are created correctly."""
    
    print("\n=== VALIDATING OUTPUT FILES ===\n")
    
    test_dirs = ['test_output_amharic', 'test_output_full']
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"✅ Directory exists: {test_dir}")
            
            train_file = os.path.join(test_dir, 'train.json')
            test_file = os.path.join(test_dir, 'test.json')
            
            if os.path.exists(train_file):
                print(f"✅ Train file exists: {train_file}")
                with open(train_file, 'r', encoding='utf-8') as f:
                    train_data = json.load(f)
                    print(f"   - Contains {len(train_data)} examples")
            else:
                print(f"❌ Train file missing: {train_file}")
            
            if os.path.exists(test_file):
                print(f"✅ Test file exists: {test_file}")
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                    print(f"   - Contains {len(test_data)} examples")
            else:
                print(f"❌ Test file missing: {test_file}")
        else:
            print(f"❌ Directory missing: {test_dir}")

def compare_with_old_output():
    """Compare output structure with old pipeline expectations."""
    
    print("\n=== COMPARING WITH OLD PIPELINE FORMAT ===\n")
    
    # Check if we have test output
    if os.path.exists('test_output_full/train.json'):
        with open('test_output_full/train.json', 'r', encoding='utf-8') as f:
            new_data = json.load(f)
        
        if new_data:
            sample = new_data[0]
            print("New pipeline output structure:")
            print(f"Keys: {list(sample.keys())}")
            print(f"Input format: {sample['input'][:100]}...")
            print(f"Output format: {sample['output'][:100]}...")
            
            # Check if structure matches expected format
            expected_keys = ['input', 'output']
            if all(key in sample for key in expected_keys):
                print("✅ Output structure matches expected format")
            else:
                print("❌ Output structure does not match expected format")
    else:
        print("❌ No test output found to compare")

def cleanup_test_files():
    """Clean up test files."""
    
    print("\n=== CLEANUP ===\n")
    
    import shutil
    test_dirs = ['test_output_amharic', 'test_output_full']
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"✅ Cleaned up: {test_dir}")
            except Exception as e:
                print(f"❌ Failed to clean up {test_dir}: {e}")

if __name__ == "__main__":
    test_small_sample()
    validate_output_files()
    compare_with_old_output()
    
    # Ask user if they want to cleanup
    response = input("\nClean up test files? (y/n): ")
    if response.lower() == 'y':
        cleanup_test_files()
    else:
        print("Test files preserved for inspection.") 