input_text = "What is the capital of France?"
output_text = "The capital of France is Paris."


if False:
    print("Using minimal prompt format")
else:
    # Use the full prompt format with instruction and response
    text = f'''<|begin_of_text|><|user|>
    {input_text}<|end_of_turn|>
    <|assistant|>
    {output_text}<|end_of_turn|>'''



import re
lines = text.splitlines()

# Check if a second line actually exists
if len(lines) > 1:
    second_line = lines[1]
    print(f"The second line is: '{second_line}'")

    if second_line.startswith(' '):
        print("✅ Yes, the second line starts with a space.")
    else:
        print("❌ No, the second line does not start with a space.")
else:
    print("The string does not have a second line.")