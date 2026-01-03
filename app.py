import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
import requests

# ================= CONFIG =================
MODEL_URL = "https://huggingface.co/Aakashsusar123/Skin_cancer_cell/resolve/main/cancer_model.h5"
MODEL_PATH = "cancer_model.h5"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
# =========================================

app = Flask(__name__)
CORS(app)

model = None


def download_model():
    if os.path.exists(MODEL_PATH):
        return

    print("⬇️ Downloading model from Hugging Face...")
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"✅ Model downloaded: {size_mb:.2f} MB")


def get_model():
    global model
    if model is None:
        download_model()
        print("🧠 Loading model...")
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded")
    return model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        img = image.load_img(tmp_path, target_size=(150, 150))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        os.remove(tmp_path)

        model = get_model()
        pred = model.predict(img)[0][0]

        return jsonify({
            "prediction": "Cancerous" if pred > 0.5 else "Benign",
            "confidence": round(float(pred), 4)
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
