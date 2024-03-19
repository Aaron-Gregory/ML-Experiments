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
sequence_length = 130
val_set_size = 500
test_set_size = 500

model_hidden_layers = 1
model_hidden_layer_size = 128

training_epochs = 100
training_batch_size = 32

dropout_fraction = 0.2

sample_seed_text = "We must never forget that"


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
    # Define the model architecture
    layers = []
    layers.append(Embedding(vocab_size, 32, input_shape=(sequence_length,)))
    for _ in range(model_hidden_layers - 1):
        layers.append(
            LSTM(
                model_hidden_layer_size,
                return_sequences=True,
                dropout=dropout_fraction,
                recurrent_dropout=dropout_fraction,
            )
        )
    layers.append(
        LSTM(
            model_hidden_layer_size,
            dropout=dropout_fraction,
            recurrent_dropout=dropout_fraction,
        )
    )
    layers.append(Dense(vocab_size, activation="softmax"))

    model = Sequential(layers)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.summary()

    print("total_params:", model.count_params())

    return model


def train_model(model, x_train, y_train, x_val, y_val):
    # Add the CSVLogger callback to save training history
    csv_logger = tf.keras.callbacks.CSVLogger("training_history.csv")

    # Train the model
    model.fit(
        x_train,
        y_train,
        epochs=training_epochs,
        batch_size=training_batch_size,
        validation_data=(x_val, y_val),
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
    # Prepping data
    x, y, vocab_size, tokenizer = prep_data()
    s1, s2 = -(val_set_size + test_set_size), -test_set_size
    x_train, x_test, x_val = x[:s1], x[s1:s2], x[s2:]
    y_train, y_test, y_val = y[:s1], y[s1:s2], y[s2:]

    print("dataset_size:", x.shape[0])
    print("training_set_size:", x_train.shape[0])
    print("vocab_size: ", vocab_size)
    print()
    print("X (train, test, val):", x_train.shape, x_test.shape, x_val.shape)
    print("y (train, test, val):", y_train.shape, y_test.shape, y_val.shape)

    # building and training model
    model = build_model(vocab_size)
    train_model(model, x_train, y_train, x_val, y_val)
    model.save("sl_model.keras")

    # evaluating results
    train_loss, train_accuracy = model.evaluate(x_train, y_train)
    print("train_loss:", train_loss)
    print("train_accuracy:", train_accuracy)

    val_loss, val_accuracy = model.evaluate(x_val, y_val)
    print("val_loss:", val_loss)
    print("val_accuracy:", val_accuracy)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("test_loss:", test_loss)
    print("test_accuracy:", test_accuracy)

    # generating sample text
    generated_text = generate_text(
        sample_seed_text, model, tokenizer, sequence_length, num_chars_to_generate=800
    )
    print(generated_text)
    with open("sample_output.txt", "w") as f:
        f.write(generated_text)
