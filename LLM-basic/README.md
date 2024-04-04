# Basic LLM Training on ArXiv Data

Welcome to the **LLM-basic** directory! This repository contains scripts and resources for training a basic Language Model (LLM) on an ArXiv dataset. The purpose of this project is to collect data from ArXiv, explore the dataset, process the collected data, and train a basic Language Model, while exploring some methodology involved with improving training and prediction performance.

## Getting Started

1. **Clone the Repository**: Clone this repository to your local machine using `git clone`.

2. **Install Dependencies**: Ensure you have all the necessary dependencies installed. I recommend setting up a venv like so:
```
python -m venv LLM_env
source LLM_env/bin/activate
pip install -r requirements.txt
```

3. **Data Collection**: Run `data_collection.py` to collect data from the ArXiv API. Ensure you have API access and configure the script accordingly. You may need to run this script multiple times with different topics in order to get a variety of training data.

4. **Explore Data**: Execute `explore_data.py` to get insights into the collected dataset. This will help you understand its size and characteristics.

5. **Process Dataset**: Run `process_dataset.py` to preprocess the dataset. This script will tokenize the `.tex` files in `dataset_raw` and save the tokenized results and tokenizer under `dataset_processed` for later use in training the Language Model.

6. **Train the Language Model**: With the dataset processed, you can now proceed to train your Language Model using the tokenized data.

## Scripts

### 1. `data_collection.py`
- **Description**: This script collects data from the ArXiv API endpoint.
- **Usage**: Run this script to collect data from ArXiv and save it for further processing.

### 2. `explore_data.py`
- **Description**: This script provides a quick look at the dataset collected, including generating a histogram of file sizes.
- **Usage**: Execute this script to explore the characteristics of the dataset collected from ArXiv.

### 3. `process_dataset.py`
- **Description**: This script processes the collected data. It pulls in all `.tex` files collected, fits a tokenizer to them, and saves the tokenizer and the tokenized files.
- **Usage**: Run this script to preprocess the dataset, tokenize the text files, and save the tokenizer for later use in training the Language Model.


Happy training! 🚀