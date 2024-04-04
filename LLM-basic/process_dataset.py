"""
This script fits a character-level tokenizer to the .tex files in dataset_raw, processes them, and
stores the resultant vocabulary and token strings under dataset_processed.
"""

import re
import tensorflow as tf
from tqdm import tqdm
import json

Tokenizer = tf.keras.preprocessing.text.Tokenizer

from data_arxiv import load_dataset
from constants import VALID_CHARACTERS, TOKENIZER_JSON_PATH, TOKENIZED_DATASET_PATH


def process_file(contents, filter_characters=False, print_stats=True):
    len_original = len(contents)

    # convert all whitespace sequences to single characters
    contents = re.sub(r"\s+", " ", contents)

    len_post_whitespace = len(contents)

    # filter out unallowed characters
    if filter_characters:
        contents = "".join(filter(lambda x: x in VALID_CHARACTERS, contents))

    len_post_filter = len(contents)

    reduction_after_whitespace = 1 - (len_post_whitespace / len_original)
    reduction_after_characters = 1 - (len_post_filter / len_original)

    if print_stats:
        print(f"Reduction after whitespace: {reduction_after_whitespace * 100:5.01f}%")
        if filter_characters:
            print(
                f"Reduction after character filter: {reduction_after_characters * 100:5.01f}%"
            )

    return contents


if __name__ == "__main__":
    text_data_arr = load_dataset()
    process_text_arr = []

    raw_length = 0
    processed_length = 0

    for text in tqdm(text_data_arr, desc="Preprocessing dataset", unit="files"):
        process_text_arr.append(process_file(text, print_stats=False))
        raw_length += len(text)
        processed_length += len(process_text_arr[-1])

    print(
        f"Text reduction due to processing: {(1 - (processed_length / raw_length)) * 100:.01f}%"
    )

    # Tokenize the text
    tokenizer = Tokenizer(char_level=True, lower=True)
    print("\nFitting tokenizer to texts (this may a while)...")
    tokenizer.fit_on_texts(process_text_arr)

    print("Saving...")
    # Save tokenizer in a way that allows for it to be reconstructed
    with open(TOKENIZER_JSON_PATH, "w") as f:
        json.dump(tokenizer.to_json(), f)

    print("Done")

    # tokenize processed dataset and save as a single file
    print("\nConverting text to sequences...")
    sequences = tokenizer.texts_to_sequences(process_text_arr)

    print("Saving...")
    with open(TOKENIZED_DATASET_PATH, "w") as f:
        json.dump({"sequences": sequences}, f)

    print("Done")