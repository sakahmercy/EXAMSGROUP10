import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained CNN model
MODEL_PATH = "image_classification.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels for Fashion-MNIST
CLASS_LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Upload folder setup
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Image Preprocessing Function
def preprocess_image(image_path):
    """Loads and preprocesses an image for Fashion-MNIST classification."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (28, 28))  # Resize to match model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=[0, -1])  # Add batch and channel dimensions
    return img

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
    file.save(filepath)

    # Preprocess and predict
    img = preprocess_image(filepath)
    prediction = model.predict(img)[0]
    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    return jsonify({"prediction": predicted_class, "confidence": confidence})

# UI for file upload
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
