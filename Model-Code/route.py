from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load models and objects
clf_binary = joblib.load("binary_model.pkl")
clf_multi = joblib.load("multi_model.pkl")
mlb = joblib.load("mlb.pkl")
model_columns = joblib.load("model_columns.pkl")

app = FastAPI(title="Student Debarment Predictor API")

# Input schema
class StudentInput(BaseModel):
    id: int
    gender: str
    math_score: int
    physics_score: int
    chemistry_score: int
    path_time_job: str
    absence_days: int
    extracurricular_activities: str
    weekly_self_study_hours: int
    career_aspiration: str
    fee_paid: int
    fee_pending: int

@app.post("/predict")
def predict(student: StudentInput):
    # Convert input to DataFrame
    X_new = pd.DataFrame([student.dict()])
    X_new = pd.get_dummies(X_new, drop_first=True)

    # Align with training features
    X_new = X_new.reindex(columns=model_columns, fill_value=0)

    # Predict debarred (Yes/No)
    debarred = clf_binary.predict(X_new)[0]

    # Predict reasons
    y_prob = clf_multi.predict_proba(X_new)
    predicted_reasons = []
    for i, probs in enumerate(y_prob):
        if probs[0][1] > 0.3:  # threshold
            predicted_reasons.append(mlb.classes_[i])

    if debarred == 1 and len(predicted_reasons) == 0:
        predicted_reasons = ["Unknown Risk"]

    return {
        "RISK": "Yes" if debarred else "No",
        "Reasons": predicted_reasons if debarred else ["None"]
    }
