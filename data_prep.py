import os
from datasets import load_dataset

# local imports
from config import config

# compute suffix for output name
percent_int = int(config["fraction"] * 100)
fraction_suffix = f"_{percent_int:03d}"

# define split string to get top fraction of the data
split_string = f"train[:{percent_int}%]"

# load top N% of the dataset in original order
print(f"Loading top {percent_int}% of '{config['data_id']}'...")
dataset = load_dataset(
    config["data_id"],
    split=split_string,
    streaming=False,
    cache_dir=config["data_cache_dir"]
)
print(f"Loaded {len(dataset)} samples.")

# build output path and save to disk
output_dir = os.path.join(config["data_dir"], config["data_name_base"] + fraction_suffix)
os.makedirs(output_dir, exist_ok=True)

print(f"Saving dataset to '{output_dir}' ...")
dataset.save_to_disk(output_dir)
print("Save complete.")
