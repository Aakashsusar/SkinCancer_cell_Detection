import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
import requests
import cv2
from werkzeug.utils import secure_filename

# ================= CONFIG =================
MODEL_URL = "https://huggingface.co/Aakashsusar123/Skin_cancer_cell/resolve/main/cancer_model.h5"
MODEL_PATH = "cancer_model_fixed.h5"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
# =========================================

app = Flask(__name__)
CORS(app)

model = None


# ================= MODEL =================
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
        print("✅ Model loaded successfully")
    return model


# ================= HELPERS =================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_skin_like_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False, "Could not read image"

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])

        color_std = np.std(rgb)

        avg_r = np.mean(rgb[:, :, 0])
        avg_g = np.mean(rgb[:, :, 1])
        avg_b = np.mean(rgb[:, :, 2])

        if avg_val < 30:
            return False, "Image too dark"

        if avg_val > 250 and avg_sat < 10:
            return False, "Image overexposed"

        if avg_b > avg_r * 1.3 or avg_g > avg_r * 1.3:
            return False, "Non-skin dominant colors"

        if not (0 <= avg_hue <= 50 or 150 <= avg_hue <= 180):
            return False, "Hue not skin-like"

        if color_std < 10 or color_std > 100:
            return False, "Insufficient skin texture"

        return True, "Valid skin-like image"

    except Exception as e:
        return False, str(e)


# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            filename = secure_filename(file.filename)
            file.save(tmp.name)
            tmp_path = tmp.name

        # Validate image
        valid, reason = is_skin_like_image(tmp_path)
        if not valid:
            os.remove(tmp_path)
            return jsonify({
                "prediction": "Invalid",
                "confidence": 0,
                "recommendation": f"Invalid image: {reason}. Please upload a clear skin image.",
                "is_invalid": True,
                "is_cancerous": False
            })

        img = image.load_img(tmp_path, target_size=(150, 150))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        os.remove(tmp_path)

        model = get_model()
        pred = float(model.predict(img)[0][0])

        malignant = pred * 100
        benign = (1 - pred) * 100
        max_conf = max(malignant, benign)

        if max_conf < 55:
            return jsonify({
                "prediction": "Invalid",
                "confidence": round(max_conf, 2),
                "recommendation": "Unclear image. Please upload a higher quality skin image.",
                "is_invalid": True,
                "is_cancerous": False
            })

        is_cancer = pred > 0.5
        result = "Malignant" if is_cancer else "Benign"

        recommendation = (
            "Please consult a dermatologist immediately."
            if is_cancer
            else "Looks benign. Continue regular skin monitoring."
        )

        return jsonify({
            "prediction": result,
            "confidence": round(malignant if is_cancer else benign, 2),
            "recommendation": recommendation,
            "is_invalid": False,
            "is_cancerous": is_cancer
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
