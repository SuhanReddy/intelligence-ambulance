from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import pandas as pd

from app.ml_pipeline import train_models, load_model_bundle, predict_patient

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
app.secret_key = "change-this-secret-key"

ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    bundle = load_model_bundle(MODEL_DIR)
    return render_template("index.html", bundle=bundle)

@app.route("/train", methods=["POST"])
def train():
    file = request.files.get("dataset")
    if not file or file.filename == "":
        flash("Please choose a CSV dataset.", "error")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Only CSV files are supported.", "error")
        return redirect(url_for("index"))

    path = DATA_DIR / secure_filename(file.filename)
    file.save(path)

    try:
        result = train_models(path, MODEL_DIR)
        flash("Models trained successfully.", "success")
        return render_template("results.html", result=result)
    except Exception as exc:
        flash(f"Training failed: {exc}", "error")
        return redirect(url_for("index"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        bundle = load_model_bundle(MODEL_DIR)
        if not bundle:
            flash("Train the models first.", "error")
            return redirect(url_for("index"))

        values = {}
        for feature in bundle["features"]:
            raw = request.form.get(feature, "").strip()
            if raw == "":
                raise ValueError(f"Missing value for: {feature}")
            values[feature] = float(raw)

        result = predict_patient(values, bundle)
        return render_template("prediction.html", result=result)
    except Exception as exc:
        flash(f"Prediction failed: {exc}", "error")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
