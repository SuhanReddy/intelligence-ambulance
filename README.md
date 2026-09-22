# The Intelligence Ambulance — AI & Human Interaction Prototype

A cleaned-up, GitHub-ready software prototype based on the **Intelligence Ambulance Bridging AI and Human Interaction Technologies** thesis/project.

The original implementation used a Tkinter desktop application, CSV upload, preprocessing, Decision Tree/Random Forest/KNN models, and a localhost socket connection between an ambulance client and hospital server. This version keeps the central ML workflow but reorganizes it into a maintainable web application with separate Python, HTML and CSS files.

## Features

- CSV dataset upload
- Automatic target-column detection (`target`, `label`, `class`, `condition`, `output`, or last column)
- Missing-value handling
- Numeric scaling and categorical encoding
- Train/test split with stratification when possible
- SVM, Random Forest and KNN comparison
- Accuracy, precision, recall and F1 metrics
- Model persistence with Joblib
- Browser-based patient prediction form
- Responsive HTML/CSS dashboard
- Clear research/educational medical disclaimer

## Project structure

```text
intelligence-ambulance/
├── run.py
├── requirements.txt
├── README.md
├── .gitignore
├── app/
│   ├── ml_pipeline.py
│   ├── templates/
│   │   ├── index.html
│   │   ├── results.html
│   │   └── prediction.html
│   └── static/
│       └── style.css
├── data/
└── models/
```

## Run locally

### 1. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

### 4. Train the models

Upload the CSV dataset used for your project. The target column should preferably be named `target`.

The application stores trained models locally in `models/`.

## Dataset expectations

The application is intentionally flexible, but the prediction form is generated from the feature columns in your CSV. For a heart-disease style dataset, a common structure is:

```text
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
```

Use the **same feature schema for training and prediction**.

## Important research note

The thesis describes SVM, Random Forest and KNN results and discusses intelligent ambulance architecture involving sensors, communication, cloud computing, AI, mobile interfaces and healthcare-system integration. This repository is a software prototype of the ML/dashboard portion; it does **not** by itself implement physical ambulance sensors, GPS routing, EHR integration, telemedicine, or a production cloud service.

Do not claim the prototype has achieved the exact percentages reported in the paper unless you reproduce those measurements using the same dataset, preprocessing, split and evaluation procedure. The dashboard calculates metrics from the dataset uploaded by the user.

## Medical safety

This is an academic/research prototype. Its predictions are not medical diagnoses and should not be used to make real clinical decisions or delay emergency care.

