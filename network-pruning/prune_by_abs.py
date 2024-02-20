import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tqdm import tqdm

BATCH_SIZE = 100


def compute_average_activations(model, data):
    avg_activations = [np.zeros(layer.units) for layer in model.layers[2:]]
    num_batches = len(data) // BATCH_SIZE + (len(data) % BATCH_SIZE != 0)

    for batch_idx in tqdm(range(num_batches)):
        batch_data = data[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        batch_activations = model(batch_data, training=False)

        for layer_idx, activation_layer in enumerate(batch_activations):
            avg_activations[layer_idx] += np.sum(np.abs(activation_layer), axis=0)

    num_examples = len(data)
    for idx, avg_activation in enumerate(avg_activations):
        avg_activations[idx] /= num_examples

    return avg_activations


def prune_neurons(model, avg_activations, threshold):
    pruned_model = tf.keras.models.clone_model(model)
    pruned_model.set_weights(model.get_weights())

    for idx, layer in enumerate(pruned_model.layers[1:]):
        mask = avg_activations[idx] >= threshold
        print(layer.get_weights())
        layer.set_weights([weight[mask] for weight in layer.get_weights()])
        layer.units = np.sum(mask)

    return pruned_model


if __name__ == "__main__":
    # Load the trained model
    model = load_model("network-pruning/mnist_model.h5")

    # Extract intermediate layers' outputs
    hidden_layer_outputs = [layer.output for layer in model.layers[1:-1]]
    activation_model = Model(inputs=model.input, outputs=hidden_layer_outputs)

    # Load and preprocess the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Compute average activations
    avg_activations = compute_average_activations(activation_model, train_images)

    # Define pruning threshold
    pruning_threshold = 0.1  # Adjust as needed

    # Prune neurons based on average activations
    pruned_model = prune_neurons(model, avg_activations, pruning_threshold)
    [1, 2][3] = 4

    # Evaluate pruned model
    pruned_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    _, test_accuracy = pruned_model.evaluate(test_images, test_labels)
    print("Test accuracy of pruned model:", test_accuracy)

    # Save pruned model
    pruned_model.save("network-pruning/pruned_mnist_model.h5")
