# process.py

import numpy as np
import tensorflow as tf

Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequential = tf.keras.models.Sequential
Embedding = tf.keras.layers.Embedding
SimpleRNN = tf.keras.layers.SimpleRNN
Dense = tf.keras.layers.Dense
LSTM = tf.keras.layers.LSTM
Dropout = tf.keras.layers.Dropout

from data_arxiv import load_dataset

# Hyperparameters
epochs = 100
batch_size = 32
sequence_length = 130
seed_text = "John: How are you, Mike?"


def prep_data():
    text_data_arr = load_dataset()

    # Tokenize the text
    tokenizer = Tokenizer(char_level=True, lower=True)
    tokenizer.fit_on_texts(text_data_arr)

    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(text_data_arr)[0]

    # Prepare input and target sequences
    input_sequences = []
    output_sequences = []

    for i in range(len(sequences) - sequence_length):
        input_sequences.append(sequences[i : i + sequence_length])
        output_sequences.append(sequences[i + sequence_length])

    input_sequences = np.array(input_sequences)
    output_sequences = np.array(output_sequences)

    vocab_size = len(tokenizer.word_index) + 1
    return input_sequences, output_sequences, vocab_size, tokenizer


def build_model(
    vocab_size,
):
    # Define the model architecture:
    model = Sequential(
        [
            Embedding(vocab_size, 32, input_shape=(sequence_length,)),
            LSTM(328, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(328, dropout=0.2, recurrent_dropout=0.2),
            Dense(vocab_size, activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.summary()

    return model


def train_model(model, x, y):
    # Add the CSVLogger callback to save training history
    csv_logger = tf.keras.callbacks.CSVLogger("training_history.csv")

    # Train the model
    model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[csv_logger],
    )


# Evaluate the model and generate text:
def generate_text(seed_text, model, tokenizer, sequence_length, num_chars_to_generate):
    generated_text = seed_text

    for _ in range(num_chars_to_generate):
        token_list = tokenizer.texts_to_sequences([generated_text])
        token_list = pad_sequences(token_list, maxlen=sequence_length, padding="pre")
        predicted_probs = model.predict(token_list, verbose=0)
        # TODO: update sampling method
        predicted_token = np.argmax(predicted_probs, axis=-1)[
            0
        ]  # Get the index of the predicted token

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                output_word = word
                break

        generated_text += output_word

    return generated_text


if __name__ == "__main__":
    x, y, vocab_size, tokenizer = prep_data()
    model = build_model(vocab_size)
    train_model(model, x, y)
    model.save("sl_model.keras")

    generated_text = generate_text(
        seed_text, model, tokenizer, sequence_length, num_chars_to_generate=800
    )
    print(generated_text)
