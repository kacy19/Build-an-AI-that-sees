import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.datasets import cifar10
import cv2

def build_transfer_model(input_shape=(96,96,3), num_classes=10):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if _name_ == "_main_":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train, y_test = y_train.squeeze(), y_test.squeeze()
    x_train = np.array([cv2.resize(img, (96,96)) for img in x_train]) / 255.0
    x_test = np.array([cv2.resize(img, (96,96)) for img in x_test]) / 255.0

    model = build_transfer_model()
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    os.makedirs("models", exist_ok=True)
    model.save("models/mobilenetv2_cifar10.h5")
