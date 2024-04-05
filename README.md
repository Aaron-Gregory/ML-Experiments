# Projects

## LLM-basic

**Overview:**
This project serves as an initial exploration into Language Model (LLM) development. The objective is to build a foundational model using a character-based tokenizer on an academic dataset.

**References:**
The initial version of this project was inspired by code available [here](https://f-a.nz/dev/develop-your-own-llm-like-chatgpt-with-tensorflow-and-keras/).

**Data Collection:**
Data was gathered from ArXiv, consisting of 11k unprocessed TeX files. This yielded 100k token sequences in the dataset. Access the dataset [here](https://drive.google.com/file/d/1RxVmHi96jF1UpUdG7HCsX6ML7B45QLKS/view?usp=sharing).

**Baseline Evaluation:**
A human baseline was established by evaluating 500 randomly selected sequences from the dataset, resulting in a 36.2% error rate.

**Best Accuracy:**
The best accuracy achieved using Long Short-Term Memory (LSTM) models resulted in a 40% error rate.

**Training Results:**
Training results are documented in [this spreadsheet](https://docs.google.com/spreadsheets/d/16Sg6wZPAjUad_ylNDI5h8d42TLUCnJUidK5uwUizjPA/edit?usp=sharing).