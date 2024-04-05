"""
Contains code relating to sequence loading, processing, and tokenization.
"""

import json
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

tokenizer_from_json = tf.keras.preprocessing.text.tokenizer_from_json

from src.constants import RAW_DATASET_DIR, TOKENIZED_DATASET_PATH, TOKENIZER_JSON_PATH


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


def save_processed_token_sequences(token_sequences):
    with open(TOKENIZED_DATASET_PATH, "w") as json_file:
        json.dump({"sequences": token_sequences}, json_file)


def load_processed_token_sequences():
    with open(TOKENIZED_DATASET_PATH, "r") as json_file:
        json_dict = json.load(json_file)

    return json_dict["sequences"]


def get_training_data(sequence_length):
    sequences = load_processed_token_sequences()

    # Prepare input and target sequences
    input_sequences = []
    output_sequences = []

    for sequence in tqdm(sequences):
        for i in range(len(sequence) - sequence_length):
            input_sequences.append(sequence[i : i + sequence_length])
            output_sequences.append(sequence[i + sequence_length])

    input_sequences = np.array(input_sequences)
    output_sequences = np.array(output_sequences)

    return input_sequences, output_sequences


def save_tokenizer(tokenizer):
    with open(TOKENIZER_JSON_PATH, "w") as json_file:
        json.dump(tokenizer.to_json(), json_file)


def load_tokenizer():
    with open(TOKENIZER_JSON_PATH, "r") as json_file:
        json_dict = json.load(json_file)

    return tokenizer_from_json(json_dict)


def get_vocab_size(tokenizer):
    """
    Assumes one padding token not found in the training sequences.
    """
    return len(tokenizer.word_index) + 1


def token_to_str(token, tokenizer):
    output_word = "<UNKNOWN>"
    for word, index in tokenizer.word_index.items():
        if index == token:
            output_word = word
            break

    return output_word


def token_sequence_to_str(input_tokens, tokenizer):
    result = ""
    for token in input_tokens:
        result += token_to_str(token, tokenizer)

    return result
