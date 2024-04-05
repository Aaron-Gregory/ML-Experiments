# Where raw and processed .tex files are stored
# That the full paths are spelled out is a bit of jank to get around GuildAI trying to copy all our training data for every run
RAW_DATASET_DIR = "/home/green/Coding/ML-Experiments/LLM-basic/dataset_raw/"
PROCESSED_DATASET_DIR = "/home/green/Coding/ML-Experiments/LLM-basic/dataset_processed/"

TOKENIZER_JSON_PATH = PROCESSED_DATASET_DIR + "tokenizer.json"
TOKENIZED_DATASET_PATH = PROCESSED_DATASET_DIR + "dataset.json"

# The set of characters that the model can use
alphabet = "abcdefghijklmnopqrstuvwxyz"
numbers = "0123456789"
special_chars = " .,!?@#$\%^&*()<>/[]{}_+-="
VALID_CHARACTERS = alphabet + alphabet.upper() + numbers + special_chars
