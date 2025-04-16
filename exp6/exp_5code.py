import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

def create_model(filter_size=3, regularization=0.001, optimizer='adam'):
    model = keras.Sequential([
        layers.Conv2D(32, (filter_size, filter_size), activation='relu', kernel_regularizer=regularizers.l2(regularization), input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (filter_size, filter_size), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularization)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

filter_sizes = [3, 5]
regularization_values = [0.001, 0.01]
batch_sizes = [32, 64]
optimizers = ['adam', 'sgd']

results = {}
for filter_size in filter_sizes:
    for reg in regularization_values:
        for batch_size in batch_sizes:
            for optimizer in optimizers:
                print(f"Training model with filter_size={filter_size}, reg={reg}, batch_size={batch_size}, optimizer={optimizer}")
                model = create_model(filter_size, reg, optimizer)
                history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
                test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
                results[(filter_size, reg, batch_size, optimizer)] = (history, test_acc)

plt.figure(figsize=(12, 6))
for key, (history, acc) in results.items():
    filter_size, reg, batch_size, optimizer = key
    plt.plot(history.history['val_accuracy'], label=f'fs={filter_size}, reg={reg}, bs={batch_size}, opt={optimizer} ({acc:.2f})')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Effect of Hyperparameters on Validation Accuracy')
plt.show()