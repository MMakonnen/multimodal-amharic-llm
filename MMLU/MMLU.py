import torch
import pandas as pd
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
import collections

def run_benchmark():
    """
    Main function to run the MMLU benchmark for Amharic.
    This version uses the question as the prompt and measures the likelihood of the full answer text.
    """
    # 1. --- Model and Tokenizer Loading ---
    print("Loading model...")
    # model_path = "finetuned_models/amharic_instruction_finetune_lr5e-05/20250619-044853_rasyosef"
    model_path = "finetuned_models/amharic_instruction_finetune_lr5e-05/20250619-183153_ours_method2/checkpoint-3472"  # Path to your merged model
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"Error loading model from path: {model_path}")
        print(f"Please ensure the path is correct. Original error: {e}")
        return

    FastLanguageModel.for_inference(model)  # Prepare the model for inference

    # 2. --- Dataset Loading ---
    print("Loading Amharic MMLU dataset from CohereLabs/Global-MMLU...")
    try:
        dataset = load_dataset("CohereLabs/Global-MMLU", "am")
        test_data = dataset['test']
    except Exception as e:
        print(f"Could not load the dataset. Please check your internet connection. Original error: {e}")
        return

    # Group questions by subject for organized evaluation
    questions_by_subject = collections.defaultdict(list)
    for item in test_data:
        questions_by_subject[item['subject']].append(item)

    # 3. --- Run Evaluation ---
    print("\nStarting Amharic MMLU benchmark (Direct Answer Likelihood)...")
    results = {}
    total_correct = 0
    total_questions = 0
    subjects = sorted(questions_by_subject.keys())

    for subject in subjects:
        subject_title = subject.replace('_', ' ').capitalize()
        print(f"\nEvaluating subject: {subject_title}")
        subject_correct = 0
        subject_questions = questions_by_subject[subject]

        for item in tqdm(subject_questions, desc=f"  {subject[:35]:<35}"):
            # The prompt is just the question, framed for a response.
            prompt = f"{item['question']}\n<|reserved_special_token_42|>\n" # Translation: "Question: ... \nAnswer:"
            choices = [item['option_a'], item['option_b'], item['option_c'], item['option_d']]
            answer_index = 0 if item['answer'] == 'A' else  \
                           1 if item['answer'] == 'B' else  \
                           2 if item['answer'] == 'C' else  \
                           3 if item['answer'] == 'D' else None

            choice_scores = []
            with torch.no_grad():
                for choice_text in choices:
                    # We evaluate the likelihood of the model generating the choice_text after the prompt.
                    full_text = f"{prompt} {choice_text}"
                    tokenized = tokenizer(full_text, return_tensors="pt")
                    input_ids = tokenized.input_ids.to(model.device)

                    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
                    prompt_input_ids = tokenized_prompt.input_ids.to(model.device)
                    # We only want to compute the loss for the choice text, not the prompt.
                    input_ids_clone = input_ids.clone()
                    input_ids_clone[0, :len(prompt_input_ids[0])] = -100  # Mask the prompt part
                    
                    # The model's loss is the average negative log-likelihood of the sequence.
                    # We use its negative as our score. This inherently normalizes for answer length.
                    outputs = model(input_ids, labels=input_ids_clone)
                    score = -outputs.loss.item()
                    choice_scores.append(score)

            prediction_index = choice_scores.index(max(choice_scores))
            if prediction_index == answer_index:
                subject_correct += 1

        # Store and print results for the subject
        accuracy = subject_correct / len(subject_questions) if subject_questions else 0
        results[subject] = {"accuracy": accuracy, "correct": subject_correct, "total": len(subject_questions)}
        total_correct += subject_correct
        total_questions += len(subject_questions)
        print(f"  Accuracy for {subject_title}: {accuracy:.4f} ({subject_correct}/{len(subject_questions)})")

    # 4. --- Display and Save Final Results ---
    overall_accuracy = total_correct / total_questions if total_questions else 0
    print("\n----------------------------------------------------")
    print(f"Overall Amharic MMLU Accuracy: {overall_accuracy:.4f}")
    print(f"Total Correct: {total_correct} / {total_questions}")
    print("----------------------------------------------------")

    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.index.name = "Subject"
    results_df = results_df.sort_values(by="accuracy", ascending=False)
    
    try:
        results_df.to_csv("amharic_mmlu_benchmark_direct_likelihood_results.csv")
        print("\nBenchmark results saved to 'amharic_mmlu_benchmark_direct_likelihood_results.csv'")
    except Exception as e:
        print(f"\nCould not save results to CSV. Error: {e}")

if __name__ == "__main__":
    run_benchmark()