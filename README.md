# Projects

## LLM-basic

This is a project to get a first intoduction to LLMs. The goal is to produce a very small foundation model on an academic dataset, with a character-based tokenizer.

First version was based on code from here: https://f-a.nz/dev/develop-your-own-llm-like-chatgpt-with-tensorflow-and-keras/

Data was collected from Arxiv - 11k unprocessed TeX files, giving token 100k sequences in the dataset.

A human baseline was evaluated across 500 sequences randomly selected from the dataset - results: 36.2% error.

Best accuracy acheived with LSTMs: 40% error.

Training are collected in this spreadsheet: https://docs.google.com/spreadsheets/d/16Sg6wZPAjUad_ylNDI5h8d42TLUCnJUidK5uwUizjPA/edit?usp=sharing