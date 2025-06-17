from unsloth import FastLanguageModel
from transformers import AutoTokenizer # Import AutoTokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_path = "test_finetuned_models/minimal_test_lr1e-05/20250618-000817" # Path to your saved PEFT checkpoint

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