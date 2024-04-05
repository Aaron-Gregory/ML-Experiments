"""
Contains code relating to sequence loading, processing, and tokenization.
"""

import os

from tqdm import tqdm
from src.constants import RAW_DATASET_DIR


def load_raw_text_sequences():
    dataset = []
    for root, _, files in os.walk(RAW_DATASET_DIR):
        for file in tqdm(files, desc="Reading in dataset", unit="files"):
            if file.endswith(".tex"):
                file_path = os.path.join(root, file)
                # Try multiple encodings
                encodings = ["utf-8", "latin-1"]  # You can extend this list as needed
                content = None
                for encoding in encodings:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            content = f.read()
                        break  # Break out of the loop if file is successfully read
                    except UnicodeDecodeError:
                        pass  # Try the next encoding
                if content is not None:
                    dataset.append(content)
                else:
                    print(f"Unable to read file: {file_path}")
    return dataset
