
from flask import Blueprint, jsonify, request
from pathlib import Path
from PIL import Image
import io

api = Blueprint("api", __name__)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE = 10 * 1024 * 1024


@api.route("/predict", methods=["POST"])
def predict():
    if "bloodImage" not in request.files:
        return jsonify({"error": "Please select an image."}), 400

    file = request.files["bloodImage"]
    screening_type = request.form.get("screeningType", "")

    if screening_type not in {"malaria", "all"}:
        return jsonify({"error": "Select a valid screening module."}), 400

    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    if file.mimetype not in ALLOWED_TYPES:
        return jsonify({"error": "Upload a JPG, PNG or WEBP image."}), 400

    data = file.read(MAX_SIZE + 1)

    if not data:
        return jsonify({"error": "The selected file is empty."}), 400

    if len(data) > MAX_SIZE:
        return jsonify({"error": "Image must be smaller than 10 MB."}), 400

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except Exception:
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    # Prototype only: the AI model is not connected yet.
    return jsonify({
        "status": "uploaded",
        "screeningType": screening_type,
        "filename": file.filename,
        "message": "Image received and validated. AI analysis is not connected yet."
    })
