import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(_, _), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
test_images = test_images / 255.0

# Load the pre-trained model
model = load_model("network-pruning/mnist_model.h5")


# Function to predict the number
def predict_number(image):
    # Reshape the image to fit the model input shape
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    # Predict the number
    prediction = model.predict(image)
    return np.argmax(prediction)


# Create a figure to hold the subplots
fig, axes = plt.subplots(3, 5, figsize=(15, 10))

# Choose 5 random indices from the test set
indices = np.random.randint(0, test_images.shape[0], size=15)

# Iterate over the subplots and plot images along with predictions
for i, ax in enumerate(axes.flat):
    # Get the image and label
    image = test_images[indices[i]]
    label = test_labels[indices[i]]

    # Predict the number
    predicted_label = predict_number(image)

    # Plot the image and display the prediction
    ax.imshow(image, cmap=plt.cm.binary)
    if label != predicted_label:
        title = f"INCORRECT: t={label}/p={predicted_label}"
    else:
        title = f"{label}"
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
