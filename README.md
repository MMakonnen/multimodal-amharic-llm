# Multimodal Amharic LLM

1. to recreate conda env locally run:
    conda env create -f env.yml
when on a cluster, try: 
    conda env create -f env_cluster.yml

2. to login using the huggingface cli (after having requested access to gated 
community models like llama), run
    huggingface-cli login
3. then run the main file to download data and train tokenizer at
    multimodal-amharic-llm/amharic_tokenizer/main.py

### Finetuning README

Run the minimal test to verify everything works:
```bash
python test_finetuning.py
```


This will:
- Check GPU and CUDA availability
- Run a 2-step training test

### Process Amharic Datasets for Finetuning

The `new_data_proc.py` script processes Amharic instruction datasets:


**Features:**
- Processes Amharic Alpaca and Dolly datasets from HuggingFace
- Creates QA data wuth  multilingual responses and translation tasks
- Generates train/test splits

**Dataset Sources:**
- `iocuydi/amharic-alpaca` - Amharic instruction-following data
- `iocuydi/amharic-dolly-15k` - Amharic question-answering data

You can consult `test_data_pipeline.py` to see how to generate the data. In general, you want to set all flags as True.

### Instruction Finetuning

### Configuration

The finetuning script `instruction_finetuning.py` has two modes:

**Test Mode** (Was used by me just to check that it works):
```python
USE_TEST_CONFIG = True  # in instruction_finetuning.py
```

**Regular Mode** (for full training):
```python
USE_TEST_CONFIG = False  # in instruction_finetuning.py
```

P.S.: I used LLM to add some comments and pretty prints in tests files (I mean, I did read them, it's not like they are misleading or false), hope it will also help
