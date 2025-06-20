# Multimodal Amharic LLM
Follow the instructions below to run the full pipeline.

## Environment Setup
```bash
conda env create -f env.yml
```

when on a cluster, try: 
```bash
conda env create -f env_cluster.yml
```

## Tokenizer
Use `amharic_tokenizer/train.py` for training.  
`amharic_tokenizer/test.py` performs a sanity test.  
- Data sources: `l-jiao/amharic-news`, `l-jiao/amharic-wikipedia`, `ljiao-amharic-common-crawl`

## Pretraining
Follow the steps in order:  
1. Modify the JSON in `config.py` to choose the datasets, model settings, lora config, training hyperparams.
1. Use `data_prep.py` to download the datasets.  
1. Launch `continued_pretraining.py`.  
- Data sources: `iocuydi/amharic-redpajama-synthetic`, `l-jiao/amharic-news`, `l-jiao/amharic-wikipedia`, `ljiao-amharic-common-crawl`

## Finetuning
Follow the steps in order:  
1. Modify the JSON in `finetune_config_full.py` to choose the training hyperparams, and set the path to the pretrained model. No need to set the dataset paths here, as they are set in `new_data_proc.py`. The lora settings cannot be changed anymore at this stage.    
1. Use `new_data_proc.py` to download the datasets.  
1. Run `instruction_finetuning.py`.  
- Data sources: `iocuydi/amharic-alpaca`, `iocuydi/amharic-dolly-15k`  

## Inference
Use `final_inference_test.py`.  

## Benchmarking
Use `MMLU/MMLU.py`.
