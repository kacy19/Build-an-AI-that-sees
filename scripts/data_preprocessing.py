import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

def load_cifar10(normalize=True):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if normalize:
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
    return x_train, y_train.squeeze(), x_test, y_test.squeeze()

def preview_samples(x, y, class_names=None, rows=2, cols=5, save_path=None):
    plt.figure(figsize=(cols*2, rows*2))
    idxs = np.random.choice(len(x), rows*cols, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(rows, cols, i+1)
        plt.imshow(x[idx])
        title = str(y[idx]) if class_names is None else class_names[y[idx]]
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if _name_ == "_main_":
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    x_train, y_train, _, _ = load_cifar10()
    os.makedirs("visualizations", exist_ok=True)
    preview_samples(x_train, y_train, classes, save_path="visualizations/day1_samples.png")
    print("Saved visualization to visualizations/day1_samples.png")
