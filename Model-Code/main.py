import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, hamming_loss

df = pd.read_csv("SIH/student_master_balanced.csv")

df["debar_risk"] = df["debar_risk"].fillna("None").astype(str)

df["reasons_list"] = df["debar_risk"].apply(
    lambda x: [] if x.strip() == "None" else [r.strip() for r in x.split(",")]
)

df["is_debarred"] = df["debar_risk"].apply(lambda x: 0 if x == "None" else 1)

mlb = MultiLabelBinarizer()
y_multi = mlb.fit_transform(df["reasons_list"])
y_binary = df["is_debarred"]

X = df.drop(columns=["debar_risk", "is_debarred", "reasons_list",
                     "first_name", "last_name", "email", "fee_due_date"])
X = pd.get_dummies(X, drop_first=True)  

# --- Binary model ---
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
clf_binary = RandomForestClassifier(n_estimators=100, random_state=42)
clf_binary.fit(X_train, y_train)
y_pred_bin = clf_binary.predict(X_test)

print("\n📊 Metrics: Binary Classification (Debarred or Not)")
print("Accuracy :", accuracy_score(y_test, y_pred_bin))
print("F1 Score :", f1_score(y_test, y_pred_bin))
print(classification_report(y_test, y_pred_bin))

# --- Multi-label model ---
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_multi, test_size=0.2, random_state=42)
clf_multi = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf_multi.fit(X_train2, y_train2)
y_pred_multi = clf_multi.predict(X_test2)

print("\n📊 Metrics: Multi-label Classification (Reasons)")
print("Hamming Loss :", hamming_loss(y_test2, y_pred_multi))
print(classification_report(y_test2, y_pred_multi, target_names=mlb.classes_))

# --- Prediction function ---
def predict_new_student(new_student):
    X_new = pd.DataFrame([new_student])
    X_new = pd.get_dummies(X_new, drop_first=True)
    X_new = X_new.reindex(columns=X.columns, fill_value=0)

    debarred = clf_binary.predict(X_new)[0]

    y_prob = clf_multi.predict_proba(X_new)
    predicted_reasons = []
    for i, probs in enumerate(y_prob):
        if probs[0][1] > 0.3:
            predicted_reasons.append(mlb.classes_[i])

    if debarred == 1 and len(predicted_reasons) == 0:
        predicted_reasons = ["Unknown Risk"]

    return debarred, predicted_reasons


# import joblib

# # Save binary model, multi-label model, encoders
# joblib.dump(clf_binary, "binary_model.pkl")
# joblib.dump(clf_multi, "multi_model.pkl")
# joblib.dump(mlb, "mlb.pkl")
# joblib.dump(X.columns, "model_columns.pkl")

# print("✅ Models saved successfully")





new_student = {
    "id": 2001,
    "gender": "Male",
    "math_score": 70,
    "physics_score": 80,
    "chemistry_score": 20,
    "path_time_job": "No",
    "absence_days": 10,
    "extracurricular_activities": "Yes",
    "weekly_self_study_hours": 6,
    "career_aspiration": "Engineer",
    "fee_paid": 100000,
    "fee_pending": 0,
}

debarred, reasons = predict_new_student(new_student)

print("\n🎯 Prediction for New Student")
print("RISK?     ->", "Yes" if debarred else "No")
print("Reasons   ->", reasons)
