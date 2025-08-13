import cv2
import numpy as np
from tensorflow.keras.models import load_model

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def predict_image(model_path, image_path, target_size=(32,32)):
    model = load_model(model_path)
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size).astype('float32')/255.0
    preds = model.predict(np.expand_dims(img_resized, axis=0))[0]
    return classes[np.argmax(preds)], np.max(preds)

if _name_ == "_main_":
    label, conf = predict_image("models/cnn_basic.h5", "assets/sample.jpg")
    print(f"Prediction: {label} ({conf:.2f})")
