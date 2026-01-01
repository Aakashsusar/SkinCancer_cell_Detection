import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify , render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
import gdown
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'cancer_model.h5'
GOOGLE_DRIVE_FILE_ID = '1hzlcYemlRXL9wxCByC4vJdR_KOGPq3MS'  # <-- Replace with your actual file ID

app = Flask(__name__)
CORS(app)


def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from Google Drive...")
        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

download_model_if_needed()

# Global variable to hold the model
model = None

def get_model():
    """Lazy load the model only when needed"""
    global model
    if model is None:
        print("Loading model... This may take a moment.")
        model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully!")
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = secure_filename(file.filename)
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        img_data = image.load_img(tmp_path, target_size=(150, 150))
        img_array = image.img_to_array(img_data) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        os.remove(tmp_path)

        current_model = get_model()
        prediction = current_model.predict(img_array)
        result = "Cancerous" if prediction[0][0] > 0.5 else "Benign"
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
