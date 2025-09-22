import os
import io
import uuid
import json
import base64
from pathlib import Path
from PIL import Image
import numpy as np

from flask import Flask, render_template, request, jsonify, send_from_directory, abort

import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import cv2

# import your modules (ensure these files are in the same directory)
from measure import CattleMeasurements
from data_manager import CattleDataManager
import predect

# ---------- Configuration ----------
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODEL_PATH = BASE_DIR / getattr(predect, "MODEL_PATH", "dog_breed_model.pth")
LABELS_FILE = BASE_DIR / getattr(predect, "LABELS_FILE", "labels.xlsx")
IMG_SIZE = getattr(predect, "IMG_SIZE", 128)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- App ----------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# ---------- Global objects (load once) ----------
data_manager = CattleDataManager(output_dir=str(BASE_DIR / "cattle_evaluations"))
measure = CattleMeasurements(calibration_factor=None)
last_prediction = None  # to store last prediction for results.html

# load labels & model
def load_labels(labels_file):
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    if str(labels_file).lower().endswith(".csv"):
        df = pd.read_csv(labels_file)
    else:
        df = pd.read_excel(labels_file)
    if 'breed' not in df.columns:
        raise ValueError("Labels file must contain a 'breed' column")
    all_breeds = sorted(df['breed'].astype(str).str.strip().unique())
    idx_to_breed = {i: b for i, b in enumerate(all_breeds)}
    return idx_to_breed

print("Loading model and labels...")
try:
    idx_to_breed = load_labels(LABELS_FILE)
    num_classes = len(idx_to_breed)
    model = predect.DogBreedCNN(num_classes=num_classes)
    state = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"Loaded model from {MODEL_PATH} with {num_classes} classes.")
except Exception as e:
    model = None
    idx_to_breed = {}
    print("Warning: model or labels couldn't be loaded at startup:", e)

# helper prediction function (re-uses predect.transform if available)
try:
    transform = predect.transform
except Exception:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def predict_topk(image_pil, topk=3):
    if model is None or not idx_to_breed:
        return []
    x = transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out[0], dim=0)
        topk = min(topk, probs.shape[0])
        top_probs, top_idxs = torch.topk(probs, k=topk)
        results = [(idx_to_breed[int(i.item())], float(p.item())) for i, p in zip(top_idxs, top_probs)]
    return results

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html", model_loaded=bool(model), num_classes=len(idx_to_breed))

@app.route("/home.html")
def home():
    return render_template("home.html")

@app.route("/upload.html")
def upload():
    return render_template("upload.html")

@app.route("/results.html")
def results_page():
    return render_template("results.html")

@app.route("/logbook.html")
def logbook():
    return render_template("logbook.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    safe = os.path.basename(filename)
    fp = UPLOAD_FOLDER / safe
    if not fp.exists():
        abort(404)
    return send_from_directory(str(UPLOAD_FOLDER), safe)

@app.route("/predict", methods=["POST"])
def predict():
    global last_prediction

    pil = None
    unique_name = None

    if "image" in request.files:  # file upload
        f = request.files["image"]
        fname = f.filename or ""
        ext = os.path.splitext(fname)[1].lower() or ".jpg"
        unique_name = f"{uuid.uuid4().hex}{ext}"
        out_path = UPLOAD_FOLDER / unique_name
        f.save(str(out_path))
        pil = Image.open(str(out_path)).convert("RGB")
    else:  # try base64 string
        data = request.get_json(force=True, silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        img_b64 = data["image"]
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(img_b64)
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            unique_name = f"{uuid.uuid4().hex}.jpg"
            pil.save(str(UPLOAD_FOLDER / unique_name))
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image: {e}"}), 400

    try:
        topk = int(request.form.get("topk", 3)) if request.form else 3
    except Exception:
        topk = 3

    calib = request.form.get("calib_cm_per_px") if request.form else None
    if calib:
        try:
            measure.calibration_factor = float(calib)
        except:
            measure.calibration_factor = None

    try:
        preds = predict_topk(pil, topk=topk)
        preds_out = [{"breed": b, "confidence": float(p)} for b, p in preds]
    except Exception as e:
        preds_out = []
        print("Prediction error:", e)

    try:
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        measurements = measure.extract_measurements(img_bgr, bbox=None)
    except Exception as e:
        print("Measurement extraction error:", e)
        measurements = None

    resp = {
        "image_url": f"/uploads/{unique_name}" if unique_name else None,
        "predictions": preds_out,
        "measurements": measurements,
        "calibration": measure.calibration_factor
    }

    last_prediction = resp
    return jsonify(resp)

@app.route("/results", methods=["GET"])
def results():
    if not last_prediction:
        return jsonify({"error": "No prediction available"}), 404
    return jsonify(last_prediction)

@app.route("/save_evaluation", methods=["POST"])
def save_evaluation():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "image_url required"}), 400

    filename = os.path.basename(image_url)
    image_path = str(UPLOAD_FOLDER / filename)
    detection = data.get("detection", {})
    measurements = data.get("measurements", {})
    classification = data.get("classification", {})
    metadata = data.get("metadata", {})

    try:
        saved_json_path = data_manager.save_evaluation(
            image_path=image_path,
            detection=detection,
            measurements=measurements,
            classification=classification,
            metadata=metadata
        )
        bpa = data_manager.export_to_bpa({
            "timestamp": saved_json_path.split("_")[-1] if saved_json_path else "",
            "measurements": measurements,
            "classification": classification
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ok", "saved_path": saved_json_path})


@app.route('/get_logbook', methods=['GET'])
def get_logbook():
    """Return all saved evaluation JSON files from cattle_evaluations folder (filesystem fallback)."""
    folder = BASE_DIR / "cattle_evaluations"
    out = []
    try:
        if not folder.exists():
            return jsonify([])

        # load files sorted by modification time (newest first)
        files = sorted(folder.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    j = json.load(f)
                    out.append(j)
            except Exception as e:
                # log and skip unreadable files
                print(f"Failed to read {p}: {e}")
        return jsonify(out)
    except Exception as e:
        print("Error in /get_logbook:", repr(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
