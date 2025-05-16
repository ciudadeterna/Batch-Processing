from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    images = data.get("images", [])
    results = []

    for img_string in images:
        img_bytes = base64.b64decode(img_string)
        processed = preprocess_image(img_bytes)
        prob = model.predict(processed)[0][0]
        results.append(float(prob))

    return jsonify({"predictions": results})
