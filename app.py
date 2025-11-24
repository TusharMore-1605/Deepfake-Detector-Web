import os
import sys
import joblib
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tempfile

# --- CONFIGURATION ---
MODEL_PATH = "CatBoost_best.joblib"
SCALER_PATH = "scaler.pkl"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'webm', 'm4a'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- LOAD MODEL & EXTRACTOR ---
try:
    # We expect feature_extractor.py to be in the same folder
    from feature_extractor import extract_features
except ImportError:
    print("FATAL ERROR: 'feature_extractor.py' not found.")
    sys.exit(1)

print("📦 Loading model + scaler...")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    if not hasattr(model, 'classes_'):
        model.classes_ = [0, 1]
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# --- HELPER FUNCTIONS ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_audio(file_path):
    """
    Core prediction logic.
    1. Resample to 16k (if needed).
    2. Extract features.
    3. Predict.
    """
    try:
        # Load audio with librosa (auto-resamples to 16000 Hz)
        # We use sr=16000 because that's what your extractor expects
        y, sr = librosa.load(file_path, sr=16000)
        
        # Save to a temporary WAV file for the extractor
        # (Your extractor requires a file path, not a numpy array)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, y, sr)
            tmp_path = tmp.name

        # Extract
        feat_vector, err = extract_features(tmp_path)
        
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        if err:
            return {"error": str(err)}

        # Predict
        feat_vector = feat_vector.reshape(1, -1)
        feat_scaled = scaler.transform(feat_vector)
        pred = model.predict(feat_scaled)[0]
        prob = model.predict_proba(feat_scaled)[0]

        # Result (Assuming 1=Original, 0=Spoof)
        is_original = (pred == 1)
        confidence = prob[1] if is_original else prob[0]
        
        return {
            "status": "success",
            "label": "Original" if is_original else "Spoof",
            "confidence": float(confidence),
            "prob_original": float(prob[1]),
            "prob_spoof": float(prob[0])
        }

    except Exception as e:
        return {"error": str(e)}

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        filename = secure_filename(file.filename)
        # If it's a blob from the microphone, it might not have an extension
        if not filename: 
            filename = "recording.wav"
            
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run Prediction
        result = predict_audio(filepath)
        
        # Cleanup uploaded file to save space
        if os.path.exists(filepath):
            os.remove(filepath)
            
        return jsonify(result)

if __name__ == '__main__':
    # host='0.0.0.0' makes it accessible on the network
    app.run(host='0.0.0.0', port=5000, debug=True)