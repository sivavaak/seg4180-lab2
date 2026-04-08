import io
import base64
import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image
from waitress import serve

from config import FLASK_SECRET_KEY
from predict import load_model, predict_single

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)


@app.route("/")
def home():
    return "Aerial House Segmentation API is running."


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image file"}), 400

    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")

    _, pred_mask = predict_single(model, img, device)

    mask_uint8 = (pred_mask * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_uint8, mode="L")
    buffer = io.BytesIO()
    mask_img.save(buffer, format="PNG")
    mask_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    coverage = float(pred_mask.mean())

    return jsonify({
        "mask_base64": mask_b64,
        "house_coverage": round(coverage, 4),
        "image_size": list(img.size),
    })


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
