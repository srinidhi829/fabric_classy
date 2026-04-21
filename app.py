from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# 🔥 Load SavedModel safely (no Keras deserialization)
loaded = tf.saved_model.load("saved_model")
infer = loaded.signatures["serve"]  # use the exported serve endpoint

class_names = ['chequered', 'paisley', 'plain', 'polka-dotted', 'striped', 'zigzagged']

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

@app.route("/")
def home():
    return "Backend Running Successfully"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    image = Image.open(file)

    x = preprocess_image(image)

    # 🔥 Call SavedModel signature
    outputs = infer(tf.constant(x))
    # get the first tensor from outputs dict
    preds = list(outputs.values())[0].numpy()

    predicted_class = class_names[int(np.argmax(preds))]
    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)