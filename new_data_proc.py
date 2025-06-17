import json
import random
from datasets import load_dataset, Dataset
import os

def process_alpaca_dataset(ds_alpaca):
    """
    Process the Amharic Alpaca dataset as-is (Amharic-only).
    Fields used:
    - prompt: Amharic instruction
    - chosen: Amharic response
    """
    new_list = []

    for item in ds_alpaca:
        amharic_prompt = item['prompt']
        amharic_response = item['chosen']

        new_list.append({
            'input': amharic_prompt,
            'output': amharic_response
        })

    print(f"Processed {len(new_list)} Alpaca examples")
    return new_list


def process_dolly_dataset(ds_dolly):
    """
    Process the Amharic Dolly dataset as-is (Amharic-only).
    Fields used:
    - instruction: Amharic instruction
    - context: Optional Amharic context
    - response: Amharic response
    """
    new_list = []

    for item in ds_dolly:
        instruction = item['instruction'].strip()
        context = item['context'].strip()
        response = item['response'].strip()

        if context:
            full_prompt = f"{context}\n\n{instruction}"
        else:
            full_prompt = instruction

        new_list.append({
            'input': full_prompt,
            'output': response
        })

    print(f"Processed {len(new_list)} Dolly examples")
    return new_list


def create_train_test_split(data_list, train_ratio=0.9):
    """Split data into train and test sets."""
    random.shuffle(data_list)
    split_idx = int(len(data_list) * train_ratio)
    return data_list[:split_idx], data_list[split_idx:]


def process_datasets_pipeline(train_ratio=0.9, output_dir='processed_amharic_data'):
    """
    Main pipeline to process the datasets and create Amharic-only training data.
    """
    print("Loading datasets...")

    ds_alpaca = load_dataset('iocuydi/amharic-alpaca')
    ds_dolly = load_dataset('iocuydi/amharic-dolly-15k')

    print(f"Loaded {len(ds_alpaca['train'])} Alpaca examples")
    print(f"Loaded {len(ds_dolly['train'])} Dolly examples")

    # Process datasets
    print("Processing Alpaca dataset...")
    alpaca_data = process_alpaca_dataset(ds_alpaca['train'])

    print("Processing Dolly dataset...")
    dolly_data = process_dolly_dataset(ds_dolly['train'])

    # Combine and split
    all_data = alpaca_data + dolly_data
    print(f"Total combined examples: {len(all_data)}")

    train_data, test_data = create_train_test_split(all_data, train_ratio)

    print(f"Train examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")

    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, 'train.json')
    test_file = os.path.join(output_dir, 'test.json')

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Saved training data to {train_file}")
    print(f"Saved test data to {test_file}")

    return train_data, test_data


def create_datasets_from_processed_data(train_data, test_data):
    """Convert processed data to HuggingFace Dataset format."""
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    return train_dataset, test_dataset


if __name__ == "__main__":
    train_data, test_data = process_datasets_pipeline(
        train_ratio=0.9,
        output_dir='processed_amharic_data'
    )

    train_dataset, test_dataset = create_datasets_from_processed_data(train_data, test_data)

    print("\nSample training example:")
    print(train_dataset[0])
