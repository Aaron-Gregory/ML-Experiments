import time
import numpy as np
import tensorflow as tf

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequential = tf.keras.models.Sequential
Embedding = tf.keras.layers.Embedding
SimpleRNN = tf.keras.layers.SimpleRNN
Dense = tf.keras.layers.Dense
LSTM = tf.keras.layers.LSTM
Dropout = tf.keras.layers.Dropout

from src.sequences import get_training_data, get_vocab_size, load_tokenizer

# Hyperparameters
sequence_length = 100
val_set_size = 1000
test_set_size = 1000

model_hidden_layers = 2
model_hidden_layer_size = 128

training_epochs = 100
training_batch_size = 32

dropout_fraction = 0.01

sample_seed_text = "We must never forget that"


def build_model(vocab_size):
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
    x, y = get_training_data(sequence_length)
    s1, s2 = -(val_set_size + test_set_size), -test_set_size
    x_train, x_test, x_val = x[:s1], x[s1:s2], x[s2:]
    y_train, y_test, y_val = y[:s1], y[s1:s2], y[s2:]

    # and tokenizer
    tokenizer = load_tokenizer()
    vocab_size = get_vocab_size(tokenizer)

    print("dataset_size:", x.shape[0])
    print("training_set_size:", x_train.shape[0])
    print("vocab_size: ", vocab_size)
    print()
    print("X (train, test, val):", x_train.shape, x_test.shape, x_val.shape)
    print("y (train, test, val):", y_train.shape, y_test.shape, y_val.shape)

    # building and training model
    model = build_model(vocab_size)

    t0 = time.time()
    train_model(model, x_train, y_train, x_val, y_val)
    training_time = time.time() - t0
    print("training_time:", training_time)

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
