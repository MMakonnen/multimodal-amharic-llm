import json
import random
from datasets import load_dataset, Dataset
import prompts


def get_prompt_no_context(language='Amharic'):
    return random.choice(prompts.PREFIX_LIST_NO_CONTEXT).format(language)

def get_prompt_with_context(language='Amharic'):
    return random.choice(prompts.PREFIX_LIST_CONTEXT).format(language)

def get_prompt_translation(src_lang, targ_lang):
    return random.choice(prompts.PREFIX_LIST_TRANSLATION).format(src_lang, targ_lang)

def get_prompt_headline_from_article():
    return random.choice(prompts.PREFIX_LIST_HEADLINE)

def get_prompt_article_from_headline():
    return random.choice(prompts.PREFIX_LIST_STORY_FROM_HEADLINE)

def get_prompt_summary_from_article():
    return random.choice(prompts.PREFIX_LIST_SUMMARY)

def get_prompt_article_from_summary():
    return random.choice(prompts.PREFIX_LIST_STORY_FROM_SUMMARY)

def process_alpaca_dataset(ds_alpaca, with_translation=False, allow_english=False, with_amharic=True):
    """
    Process the Amharic Alpaca dataset from datasets library as standalone Amharic Q&A pairs.
    No cross-lingual pairing since reference_index doesn't correspond to valid English data.
    
    Dataset structure:
    - prompt: Amharic instruction/prompt
    - chosen: Amharic response
    - error_suspicion: boolean flag
    - reference_index: original index (not reliable for pairing)
    """
    
    new_list = []
    
    for item in ds_alpaca:
        prompt_format = get_prompt_no_context() + "\nHuman: {}\nAssistant{}: "
        
        amharic_prompt = item['prompt']
        amharic_response = item['chosen']
        
        # Treat as standalone Amharic question-answer pairs
        if with_amharic:
            # Amharic prompt to Amharic response
            new_list.append({
                'input': prompt_format.format(amharic_prompt, ' [Amharic] '), 
                'output': amharic_response
            })
        
    return new_list

def process_dolly_dataset(ds_dolly, with_translation=True, allow_english=True, with_amharic=True):
    """
    Process the Amharic Dolly dataset from datasets library.
    
    Dataset structure:
    - category: task category
    - instruction: Amharic instruction/question
    - context: Amharic context (can be empty)
    - response: Amharic response
    - reference_response: original English response
    """
    
    new_list = []
    
    for item in ds_dolly:
        prompt_format_no_context = get_prompt_no_context() + "\nHuman: {}\nAssistant{}: "
        prompt_format_context = get_prompt_with_context() + "\nContext: {}\nHuman: {}\nAssistant{}: "
        prompt_format_translation_eng_am = get_prompt_translation('English', 'Amharic') + "\nEnglish: {}\nAmharic: "
        prompt_format_translation_am_eng = get_prompt_translation('Amharic', 'English') + "\nAmharic: {}\nEnglish: "
        
        amharic_instruction = item['instruction']
        amharic_response = item['response']
        amharic_context = item['context']
        english_response = item['reference_response']
        
        # Process based on whether context is available
        if len(amharic_context.strip()) > 0:
            if with_amharic:
                # Amharic context + instruction -> Amharic response
                new_list.append({
                    'input': prompt_format_context.format(amharic_context, amharic_instruction, ' [Amharic] '),
                    'output': amharic_response
                })
            
            if allow_english:
                # Amharic context + instruction -> English response
                new_list.append({
                    'input': prompt_format_context.format(amharic_context, amharic_instruction, ' [English] '),
                    'output': english_response
                })
        else:
            if with_amharic:
                # Amharic instruction -> Amharic response
                new_list.append({
                    'input': prompt_format_no_context.format(amharic_instruction, ' [Amharic] '),
                    'output': amharic_response
                })
            
            if allow_english:
                # Amharic instruction -> English response
                new_list.append({
                    'input': prompt_format_no_context.format(amharic_instruction, ' [English] '),
                    'output': english_response
                })
        
        if with_translation:
            # Translation pairs for responses
            new_list.append({
                'input': prompt_format_translation_am_eng.format(amharic_response),
                'output': english_response
            })
            new_list.append({
                'input': prompt_format_translation_eng_am.format(english_response),
                'output': amharic_response
            })
    
    return new_list

def create_train_test_split(data_list, train_ratio=0.9):
    """Split data into train and test sets."""
    random.shuffle(data_list)
    split_idx = int(len(data_list) * train_ratio)
    return data_list[:split_idx], data_list[split_idx:]

def process_datasets_pipeline(
    with_translation=True,
    allow_english=True, 
    with_amharic=True,
    train_ratio=0.9,
    output_dir='processed_data'
):
    """
    Main pipeline to process the datasets and create training data.
    """
    
    print("Loading datasets...")
    
    # Load datasets from HuggingFace
    ds_alpaca = load_dataset('iocuydi/amharic-alpaca')
    ds_dolly = load_dataset('iocuydi/amharic-dolly-15k')
    
    print(f"Loaded {len(ds_alpaca['train'])} Alpaca examples")
    print(f"Loaded {len(ds_dolly['train'])} Dolly examples")
    
    # Process Alpaca dataset
    print("Processing Alpaca dataset...")
    alpaca_processed = process_alpaca_dataset(
        ds_alpaca['train'], 
        with_translation=with_translation,
        allow_english=allow_english,
        with_amharic=with_amharic
    )
    
    # Process Dolly dataset  
    print("Processing Dolly dataset...")
    dolly_processed = process_dolly_dataset(
        ds_dolly['train'],
        with_translation=with_translation,
        allow_english=allow_english,
        with_amharic=with_amharic
    )
    
    print(f"Generated {len(alpaca_processed)} Alpaca training examples")
    print(f"Generated {len(dolly_processed)} Dolly training examples")
    
    # Combine all datasets
    all_data = alpaca_processed + dolly_processed
    print(f"Total examples: {len(all_data)}")
    
    # Split into train and test
    train_data, test_data = create_train_test_split(all_data, train_ratio)
    
    print(f"Train examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    
    # Save processed data
    import os
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
    """
    Convert processed data back to HuggingFace Dataset format.
    """
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    # Example usage
    train_data, test_data = process_datasets_pipeline(
        with_translation=True,
        allow_english=True,
        with_amharic=True,
        train_ratio=0.9,
        output_dir='processed_amharic_data'
    )
    
    # Optionally convert back to Dataset format
    train_dataset, test_dataset = create_datasets_from_processed_data(train_data, test_data)
    
    print("\nSample training example:")
    print(train_dataset[0]) 