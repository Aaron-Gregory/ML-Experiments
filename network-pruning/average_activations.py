import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Model
import numpy as np

# Load the MNIST dataset
(_, _), (test_images, _) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
test_images = test_images / 255.0

# Load the pre-trained model
model = load_model("network-pruning/mnist_model.h5")

# Extract intermediate layers' outputs
layer_outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
activation_model = Model(inputs=model.input, outputs=layer_outputs)


# Function to compute activations for a batch of images
def compute_activations(images):
    activations = activation_model.predict(images)
    return activations


# Function to compute the average activation value for each layer
def compute_average_activations():
    batch_activations = compute_activations(test_images)
    average_activations = [
        np.mean(activations, axis=0) for activations in batch_activations
    ]

    return average_activations


# Compute the average activation values
average_activations = compute_average_activations()

# Print the average activation values for each layer
for i, activation in enumerate(average_activations):
    print(f"Layer {i + 1}:")
    print(activation)
    print()
