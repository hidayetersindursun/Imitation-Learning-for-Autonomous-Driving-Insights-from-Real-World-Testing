import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from _dataset_tf import CustomDataset, compute_mean_std

# Check if GPU is available
device = "GPU" if tf.config.experimental.list_physical_devices("GPU") else "CPU"
print("Using device:", device)

# Load test data
test_path = 'data/testData.csv'
test_dataset = CustomDataset(test_path)

# Compute mean and std deviation
print("Computing Mean and Std Deviation...")
mean, std = compute_mean_std(test_dataset)
print("Computed Mean:", mean)
print("Computed Std:", std)

# Load trained model
model = tf.saved_model.load('model')

def evaluate_model(model, test_dataset):
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=2)
    print(f'Test Loss: {test_loss:.6f}')
    print(f'Test Accuracy: {test_accuracy:.6f}')
    return test_loss, test_accuracy

def plot_predictions(true_labels, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(true_labels, color='blue', label='True Values')
    plt.plot(predictions, color='red', label='Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('True vs Predicted Values')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Evaluate the model
    true_labels = []
    predictions = []

    for images, labels in test_dataset:

        # permute images to match model input shape
        images = tf.transpose(images, perm=[0, 3, 1, 2])
        images = tf.cast(images, tf.float32)
        preds = model(images)
        true_labels.extend(labels.numpy())
        predictions.extend(preds.numpy())

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    test_loss, test_accuracy = evaluate_model(model, test_dataset)
    plot_predictions(true_labels, predictions)
