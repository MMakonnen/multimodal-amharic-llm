from transformers import pipeline

llama3_am = pipeline(
    "text-generation",
    model="rasyosef/Llama-3.2-1B-Amharic-Instruct",
    device_map="auto"
  )

messages = [{"role": "user", "content": "ሦስት የአውሮፓ አገሮችን ጥቀስ"}]
print(llama3_am(messages, max_new_tokens=128, repetition_penalty=1.05, return_full_text=False))
