from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

BUNDLE_FILE = "model_bundle.joblib"

def _detect_target(df):
    candidates = ["target", "label", "class", "condition", "output"]
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return df.columns[-1]

def _prepare_target(series):
    if series.isna().any():
        series = series.fillna(series.mode().iloc[0])
    encoder = LabelEncoder()
    y = encoder.fit_transform(series.astype(str))
    return y, encoder

def _build_preprocessor(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))

    if not transformers:
        raise ValueError("The dataset must contain at least one usable feature column.")
    return ColumnTransformer(transformers=transformers), numeric, categorical

def train_models(csv_path, model_dir):
    df = pd.read_csv(csv_path)
    if df.shape[0] < 20:
        raise ValueError("Dataset is too small. Provide at least 20 rows.")
    df = df.drop_duplicates().reset_index(drop=True)

    target = _detect_target(df)
    if df[target].nunique(dropna=True) < 2:
        raise ValueError(f"Target column '{target}' must contain at least two classes.")

    X = df.drop(columns=[target])
    y, label_encoder = _prepare_target(df[target])

    # Keep columns with at least one non-null value.
    X = X.dropna(axis=1, how="all")
    preprocessor, numeric, categorical = _build_preprocessor(X)

    class_counts = pd.Series(y).value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
    )

    models = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(n_neighbors=min(5, max(1, len(X_train)//10)))
    }

    results = {}
    trained = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        results[name] = {
            "accuracy": round(accuracy_score(y_test, pred) * 100, 2),
            "precision": round(precision_score(y_test, pred, average="weighted", zero_division=0) * 100, 2),
            "recall": round(recall_score(y_test, pred, average="weighted", zero_division=0) * 100, 2),
            "f1": round(f1_score(y_test, pred, average="weighted", zero_division=0) * 100, 2),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist()
        }
        trained[name] = pipe

    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    joblib.dump(trained, model_dir / "models.joblib")
    joblib.dump(label_encoder, model_dir / "label_encoder.joblib")

    metadata = {
        "dataset": Path(csv_path).name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target": target,
        "features": X.columns.tolist(),
        "classes": label_encoder.classes_.tolist(),
        "results": results
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata

def load_model_bundle(model_dir):
    model_dir = Path(model_dir)
    meta = model_dir / "metadata.json"
    models = model_dir / "models.joblib"
    encoder = model_dir / "label_encoder.joblib"
    if not (meta.exists() and models.exists() and encoder.exists()):
        return None
    metadata = json.loads(meta.read_text())
    metadata["models"] = joblib.load(models)
    metadata["label_encoder"] = joblib.load(encoder)
    return metadata

def predict_patient(values, bundle):
    row = pd.DataFrame([values], columns=bundle["features"])
    predictions = {}
    for name, model in bundle["models"].items():
        encoded = int(model.predict(row)[0])
        label = bundle["label_encoder"].inverse_transform([encoded])[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(row)[0]) * 100)
        predictions[name] = {
            "class": str(label),
            "confidence": round(confidence, 2) if confidence is not None else None
        }

    # This is a research/demo consensus only; it is not a medical diagnosis.
    labels = [item["class"] for item in predictions.values()]
    consensus = max(set(labels), key=labels.count)
    return {"predictions": predictions, "consensus": consensus}
