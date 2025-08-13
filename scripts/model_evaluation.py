import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    plt.savefig(save_path)

if _name_ == "_main_":
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test = x_test.astype('float32')/255.0
    y_test = y_test.squeeze()

    model = load_model("models/cnn_basic.h5")
    y_pred = np.argmax(model.predict(x_test), axis=1)

    print(classification_report(y_test, y_pred, target_names=classes))
    os.makedirs("visualizations", exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, classes, "visualizations/day3_confusion_matrix.png")
