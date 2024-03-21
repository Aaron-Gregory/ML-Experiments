"""
This script is for getting the data on which I'll evaluate a human benchmark. Output goes into two files containing xs and ys.
"""

import numpy as np
from process import prep_data

NUM_EXAMPLES = 500


def token_to_str(token, tokenizer):
    output_word = "UNKNOWN"
    for word, index in tokenizer.word_index.items():
        if index == token:
            output_word = word
            break

    return output_word


def input_to_str(input_tokens, tokenizer):
    result = ""
    for token in input_tokens:
        result += token_to_str(token, tokenizer)

    return result


if __name__ == "__main__":
    x, y, vocab_size, tokenizer = prep_data()
    print(x.shape)
    xs, ys = [], []
    for _ in range(NUM_EXAMPLES):
        index = np.random.randint(0, x.shape[0])
        xs.append(input_to_str(x[index], tokenizer).replace("\n", "\\n"))
        ys.append(token_to_str(y[index], tokenizer).replace("\n", "\\n"))

    with open("out_x.txt", "w") as f:
        for s in xs:
            f.write(s + "\n")

    with open("out_y.txt", "w") as f:
        for s in ys:
            f.write(s + "\n")
