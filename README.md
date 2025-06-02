# multimodal-amharic-llm

Setup:

1. to recreate conda env locally run:
    conda env create -f env.yml
2. to login using the huggingface cli (after having requested access to gated community models like llama), run
    huggingface-cli login
3. then run the main file to download data and train tokenizer at
    multimodal-amharic-llm/amharic_tokenizer/main.py