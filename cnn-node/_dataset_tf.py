import tensorflow as tf
import pandas as pd
import numpy as np

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, csv_file, batch_size=1, shuffle=True):
        self.data = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch_data = self.data.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]

        images = []
        labels = []

        for _, row in batch_data.iterrows():
            img_path = row['testImagePaths']
            label = row['testLabels']

            # Load and preprocess image
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(672, 224))
            img = tf.keras.preprocessing.image.img_to_array(img)

            images.append(img)
            labels.append(label)

        return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)


# Compute mean and standard deviation from your dataset for images
def compute_mean_std(dataset):
    mean = 0.0
    std = 0.0
    num_samples = len(dataset)

    for images, _ in dataset:
        # Normalize the images to [0, 1]
        images = images / 255.0

        # Compute mean across height, width, and channel dimensions
        mean += tf.reduce_mean(images, axis=(0, 1, 2))
        
        # Compute standard deviation across height, width, and channel dimensions
        std += tf.math.reduce_std(images, axis=(0, 1, 2))

    mean /= num_samples
    std /= num_samples

    return mean, std