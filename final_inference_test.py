from unsloth import FastLanguageModel


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "finetuned_models/amharic_instruction_finetune_lr5e-05", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# _alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# {}

# ### Response:
# {}"""

# # Becomes:
# alpaca_prompt = """ከዚህ በታች አንድን ተግባር የሚገልጽ መመሪያ አለ. ጥያቄውን በትክክል የሚያጠናቅቅ ምላሽ ይጻፉ.

# ### መመሪያ:
# {}

# ### ምላሽ:
# {}"""

inputs = tokenizer(
[
    '''
    ጨረቃን በዝርዝር ግለጽ።
    '''
], return_tensors = "pt").to("cuda")


from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   repetition_penalty = 1.1, do_sample = True, temperature = 0.7)