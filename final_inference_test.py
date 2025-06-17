from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

# Proper chat prompt
# prompt = """The man walked down the street, and he saw a dog. The dog was very friendly and """
prompt = "Hello."

text = f'''<|begin_of_text|><|user|>
            {prompt}<|end_of_text|>
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
