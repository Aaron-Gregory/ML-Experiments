import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(_, _), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
test_images = test_images / 255.0

# Load the pre-trained model
model = load_model("network-pruning/mnist_model.h5")

# Extract intermediate layers' outputs
layer_outputs = [layer.output for layer in model.layers[1:]]  # Exclude the input layer
activation_model = Model(inputs=model.input, outputs=layer_outputs)


# Function to predict the number and get intermediate layer outputs
def predict_number_with_activations(image):
    # Reshape the image to fit the model input shape
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    # Predict the number
    activations = activation_model.predict(image)
    prediction = model.predict(image)
    return np.argmax(prediction), activations


# Plot the original images and their activations
plt.figure(figsize=(15, 10))

for row in range(3):
    # Choose a random image from the test set
    index = np.random.randint(0, test_images.shape[0])
    image = test_images[index]
    label = test_labels[index]

    # Predict the number and get intermediate layer outputs
    predicted_label, activations = predict_number_with_activations(image)

    plt.subplot(3, 5, 1 + row * 5)
    plt.imshow(image, cmap=plt.cm.binary)
    if row == 0:
        plt.title(f"Original Image")
    plt.axis("off")

    # Plot the activations for each layer
    for i, activation in enumerate(activations):
        if len(activation.shape) > 2:
            activation_grid = np.sum(activation[0], axis=-1)
        else:
            activation_grid = activation[0]
        plt.subplot(3, 5, i + 2 + row * 5)
        plt.imshow([activation_grid], cmap="viridis", aspect="auto")
        if row == 0:
            plt.title(f"Layer {i + 1}")
        plt.axis("off")

plt.tight_layout()
plt.show()
