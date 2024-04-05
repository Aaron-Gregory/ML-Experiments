import os
import matplotlib.pyplot as plt
from src.constants import RAW_DATASET_DIR


# Function to get file sizes
def get_file_sizes(directory):
    file_sizes = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)  # in bytes
                file_sizes.append(file_size)
    return file_sizes


if __name__ == "__main__":
    # Get file sizes
    file_sizes = get_file_sizes(RAW_DATASET_DIR)

    # Calculate various metrics
    average_file_size = sum(file_sizes) / len(file_sizes)  # in bytes
    max_file_size = max(file_sizes)  # in bytes
    min_file_size = min(file_sizes)  # in bytes

    # Print metrics
    print("Dataset Metrics:")
    print("Number of files:", len(file_sizes))
    print("Average file size:", round(average_file_size), "bytes")
    print("Maximum file size:", max_file_size, "bytes")
    print("Minimum file size:", min_file_size, "bytes")

    # Plot distribution of file sizes
    plt.hist(file_sizes, bins=20, color="skyblue", edgecolor="black")
    plt.xlabel("File Size (bytes)")
    plt.ylabel("Count")
    plt.title("Distribution of File Sizes")
    plt.grid(True)
    plt.show()
