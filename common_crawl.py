import os
import requests
from warcio.archiveiterator import ArchiveIterator
import fasttext
from tqdm import tqdm
import gzip
from collections import defaultdict

INDEX_URL = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-21/wet.paths.gz"
BASE_URL = "https://data.commoncrawl.org/"
FASTTEXT_MODEL = "lid.176.bin"
AMHARIC_CODE = "__label__am"
RAW_TEXT_DIR = "raw_text"
WET_TEMP_DIR = "temp_wet_files"
LOG_FILE = "processed_wet_files.log"

# Create directories if they don't exist
os.makedirs(RAW_TEXT_DIR, exist_ok=True)
os.makedirs(WET_TEMP_DIR, exist_ok=True)

def get_wet_paths(index_url):
    resp = requests.get(index_url, stream=True)
    resp.raise_for_status()
    wet_paths = []
    with gzip.GzipFile(fileobj=resp.raw) as f:
        for line in f:
            wet_paths.append(line.decode('utf-8').strip())
    return wet_paths

def download_file(url, dest):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def extract_amharic_from_wet(wet_path, fasttext_model, base_output_name):
    model = fasttext.load_model(fasttext_model)
    amharic_count = 0
    total_checked = 0
    lang_counter = defaultdict(int)

    with open(wet_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'conversion':
                text = record.content_stream().read().decode('utf-8', errors='ignore')
                if text.strip():
                    label, _ = model.predict(text.replace('\n', ' ')[:1000])
                    lang_counter[label[0]] += 1
                    total_checked += 1
                    # print(f"Detected language: {label[0]}")
                    if label[0] == AMHARIC_CODE:
                        file_name = f"{base_output_name}_{amharic_count:05d}.txt"
                        output_path = os.path.join(RAW_TEXT_DIR, file_name)
                        with open(output_path, 'w', encoding='utf-8') as out_f:
                            out_f.write(text)
                        print(f"✓ Amharic text saved: {file_name}")
                        amharic_count += 1

    print(f"--- Summary for {wet_path} ---")
    print(f"Total texts checked: {total_checked}")
    print(f"Amharic texts found: {amharic_count}")
    print("Language distribution:")
    # for lang, count in lang_counter.items():
        # print(f"  {lang}: {count}")
    if total_checked > 0:
        print(f"Amharic ratio: {amharic_count / total_checked:.2%}")

    return amharic_count

def load_processed_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def append_to_log(entry):
    with open(LOG_FILE, 'a') as f:
        f.write(entry + "\n")

def main():
    processed_files = load_processed_log()
    wet_paths = get_wet_paths(INDEX_URL)
    print(f"Found {len(wet_paths)} WET files.")

    model_path = FASTTEXT_MODEL
    if not os.path.exists(model_path):
        print(f"Please download the fastText language ID model and save it as {model_path}")
        print("Download link: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
        return

    for wet_rel_path in tqdm(wet_paths, desc="Processing WET files"):
        wet_filename = os.path.basename(wet_rel_path)
        if wet_filename in processed_files:
            print(f"Skipping already processed file: {wet_filename}")
            continue

        wet_url = BASE_URL + wet_rel_path
        wet_filepath = os.path.join(WET_TEMP_DIR, wet_filename)
        base_output_name = wet_filename.replace('.warc.wet.gz', '').replace('.', '_')

        print(f"\nDownloading {wet_url} ...")
        download_file(wet_url, wet_filepath)

        print(f"Extracting Amharic texts from {wet_filename} ...")
        count = extract_amharic_from_wet(wet_filepath, model_path, base_output_name)

        print(f"✓ Completed extraction from {wet_filename}. Amharic texts: {count}")
        os.remove(wet_filepath)
        print(f"✗ Deleted temp file {wet_filepath}")
        append_to_log(wet_filename)

if __name__ == "__main__":
    main()