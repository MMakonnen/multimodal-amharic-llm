from unsloth import FastLanguageModel

model_path = "finetuned_models/amharic_instruction_finetune_lr5e-05/20250618-012702_rasyosef_pretrained_8epochs_finetuned" # Path to your saved PEFT checkpoint

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)


embedding = model.get_input_embeddings()
print(f"Model embedding dimensions: {embedding.weight.shape}")

FastLanguageModel.for_inference(model)
# Proper chat prompt
prompt = "ፕላኔቷን ምድር ግለጽ።"

text = "<|begin_of_text|><|user|>\n{input_text}\n<|end_of_turn|><|assistant|>\n"

inputs = tokenizer([text], return_tensors="pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=1024*10,
    repetition_penalty=1.5,
    do_sample=True,
    top_k=8,
    top_p=0.8,
    temperature=0.5
)