from unsloth import FastLanguageModel

model_path = "trainer_output/checkpoint-96500" # Path to your saved PEFT checkpoint

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)
# Proper chat prompt
prompt = "ሦስት የአውሮፓ አገሮችን ጥቀስ"

text = f"{prompt}"

inputs = tokenizer([text], return_tensors="pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=1024,
    repetition_penalty=1.2,
    do_sample=True,
    top_k=8,
    top_p=0.8,
    temperature=0.5,
    eos_token_id=tokenizer.eos_token_id,

)