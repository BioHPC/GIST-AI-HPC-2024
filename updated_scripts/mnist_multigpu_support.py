import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import argparse
import time

# Define the strategy
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Start time measurement
start_time = tf.timestamp()

# Load and preprocess the MNIST dataset
def load_and_preprocess_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    
    train_images = train_images[..., tf.newaxis].astype('float32') / 255.0
    test_images = test_images[..., tf.newaxis].astype('float32') / 255.0
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    
    return train_dataset, test_dataset

# Prepare the dataset
def prepare_dataset(dataset, batch_size):
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Define the model architecture
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on MNIST dataset with MirroredStrategy.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    # Load and preprocess the data
    train_dataset, test_dataset = load_and_preprocess_data()
    
    # Prepare the datasets
    train_dataset = prepare_dataset(train_dataset, batch_size)
    test_dataset = prepare_dataset(test_dataset, batch_size)

    # Build the model within the strategy scope
    with strategy.scope():
        model = create_model()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    
    # Train the model
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    # End time measurement
    end_time = tf.timestamp()
    final_time = end_time - start_time
    print(f"Total time taken: {final_time} seconds")

