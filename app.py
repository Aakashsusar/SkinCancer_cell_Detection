import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
import gdown
from werkzeug.utils import secure_filename

# ================= CONFIG =================
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = "cancer_model.h5"
GOOGLE_DRIVE_FILE_ID = "1hzlcYemlRXL9wxCByC4vJdR_KOGPq3MS"
EXPECTED_MODEL_SIZE_MB = 120  # sanity check
# =========================================

app = Flask(__name__)
CORS(app)

model = None


# ---------- MODEL DOWNLOAD ----------
def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        if size_mb > EXPECTED_MODEL_SIZE_MB:
            print("✅ Model already present and valid")
            return
        else:
            print("⚠️ Corrupted model detected. Re-downloading...")
            os.remove(MODEL_PATH)

    print("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

    gdown.download(
        url,
        MODEL_PATH,
        quiet=False,
        fuzzy=True
    )

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("❌ Model download failed")

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if size_mb < EXPECTED_MODEL_SIZE_MB:
        os.remove(MODEL_PATH)
        raise RuntimeError("❌ Model download incomplete")

    print(f"✅ Model downloaded successfully ({size_mb:.2f} MB)")


# ---------- MODEL LOAD ----------
def get_model():
    global model
    if model is None:
        download_model_if_needed()
        print("🧠 Loading TensorFlow model...")
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded")
    return model


# ---------- UTILS ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------- ROUTES ----------
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
            return jsonify({"error": "Empty file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        img = image.load_img(tmp_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        os.remove(tmp_path)

        model = get_model()
        prediction = model.predict(img_array)[0][0]

        result = "Cancerous" if prediction > 0.5 else "Benign"

        return jsonify({
            "prediction": result,
            "confidence": round(float(prediction), 4)
        })

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
