import os
import requests
import tempfile
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# ================= CONFIG =================
MODEL_PATH = "cancer_model.h5"
FILE_ID = "1hzlcYemlRXL9wxCByC4vJdR_KOGPq3MS"
DOWNLOAD_URL = "https://drive.google.com/uc?export=download"
MIN_MODEL_MB = 50
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
# =========================================

app = Flask(__name__)
CORS(app)

model = None


def download_model():
    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists")
        return

    print("⬇️ Downloading model from Google Drive (safe method)...")

    session = requests.Session()
    response = session.get(DOWNLOAD_URL, params={"id": FILE_ID}, stream=True)

    # Handle confirmation token
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(
                DOWNLOAD_URL,
                params={"id": FILE_ID, "confirm": value},
                stream=True,
            )

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"📦 Model size: {size_mb:.2f} MB")

    if size_mb < MIN_MODEL_MB:
        os.remove(MODEL_PATH)
        raise RuntimeError("❌ Model download failed (incomplete file)")


def get_model():
    global model
    if model is None:
        download_model()
        print("🧠 Loading TensorFlow model...")
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")
    return model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            path = tmp.name

        img = image.load_img(path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        os.remove(path)

        model = get_model()
        pred = model.predict(img_array)

        result = "Cancerous" if pred[0][0] > 0.5 else "Benign"
        return jsonify({"prediction": result})

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
