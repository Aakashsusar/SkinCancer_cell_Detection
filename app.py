import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
import gdown
from werkzeug.utils import secure_filename

# ---------------- CONFIG ----------------

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = "cancer_model.h5"
GOOGLE_DRIVE_FILE_ID = "1hzlcYemlRXL9wxCByC4vJdR_KOGPq3MS"

# ----------------------------------------

app = Flask(__name__)
CORS(app)

model = None  # Lazy-loaded model


# ---------------- HELPERS ----------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model():
    """
    Download model from Google Drive if missing
    and load it only once (lazy loading).
    """
    global model

    if model is None:
        if not os.path.exists(MODEL_PATH):
            print("📥 Downloading model from Google Drive...")
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

            gdown.download(
                url,
                MODEL_PATH,
                quiet=False,
                fuzzy=True,
                use_cookies=False
            )

            print("✅ Model downloaded successfully")

        print("🧠 Loading TensorFlow model...")
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded")

    return model


# ---------------- ROUTES ----------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            filename = secure_filename(file.filename)
            file.save(tmp.name)
            tmp_path = tmp.name

        img = image.load_img(tmp_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        os.remove(tmp_path)

        current_model = get_model()
        prediction = current_model.predict(img_array)

        result = "Cancerous" if prediction[0][0] > 0.5 else "Benign"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- ENTRY ----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
