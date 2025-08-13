import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def build_basic_cnn(input_shape=(32,32,3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history, save_prefix):
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title('Model Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.savefig(f"{save_prefix}_acc.png"); plt.close()

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.savefig(f"{save_prefix}_loss.png"); plt.close()

if _name_ == "_main_":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
    y_train, y_test = y_train.squeeze(), y_test.squeeze()
    model = build_basic_cnn()
    history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    model.save("models/cnn_basic.h5")
    plot_history(history, "visualizations/day2_cnn_history")
