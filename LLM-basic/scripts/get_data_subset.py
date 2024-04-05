"""
This script is for getting the data on which a human benchmark can be evaluated. Output goes into two files containing xs and ys.
"""

import numpy as np
from train_model import sequence_length
from src.sequences import (
    get_training_data,
    load_tokenizer,
    token_sequence_to_str,
    token_to_str,
)

NUM_EXAMPLES = 500

if __name__ == "__main__":
    x, y = get_training_data(sequence_length)
    tokenizer = load_tokenizer()
    print(x.shape)
    xs, ys = [], []
    for _ in range(NUM_EXAMPLES):
        index = np.random.randint(0, x.shape[0])
        xs.append(token_sequence_to_str(x[index], tokenizer).replace("\n", "\\n"))
        ys.append(token_to_str(y[index], tokenizer).replace("\n", "\\n"))

    with open("out_x.txt", "w") as f:
        for s in xs:
            f.write(s + "\n")

    with open("out_y.txt", "w") as f:
        for s in ys:
            f.write(s + "\n")
