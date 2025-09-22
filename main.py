################################################################
#                       Imports                                #
################################################################
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import List
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import joblib
from study_planner import result,StudyPlan
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


from report import generate_pdf_report
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


cred = credentials.Certificate("sih25-9d8bb-firebase-adminsdk-fbsvc-bf3e409ea5.json")  # download from Firebase console
firebase_admin.initialize_app(cred)

db = firestore.client()


################################################################
#                   State Definitions                          #
################################################################

class Student(BaseModel):
    studentid: str=Field(min_length=5,max_length=6)
    name:str
    attendance:int
    aggregate:int=Field(gt=0,lt=100)
    feeStatus:str
    riskLevel:str

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


class StudyPlanRequest(BaseModel):
    comments: str
    email: str


app=FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

clf_binary = joblib.load(r"Saved-Models\binary_model.pkl")
clf_multi = joblib.load(r"Saved-Models\multi_model.pkl")
mlb = joblib.load(r"Saved-Models\mlb.pkl")
model_columns = joblib.load(r"Saved-Models\model_columns.pkl")

SENDER_EMAIL = os.getenv("GMAIL_ADDRESS")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dotenv import load_dotenv

def send_email_gmail(recipient_email: str, comments: str, studyplan: str, pdf_path: str = None):
    """
    Sends study plan + comments via Gmail SMTP.
    Optionally attaches a PDF report if pdf_path is provided.
    """
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    msg["Subject"] = "Personalized Study Plan Report"

    # Email body
    body = f"""
    Dear Parent/Guardian,

    We hope this message finds you well.

    As part of our ongoing commitment to support every student’s academic journey, 
    we are sharing the mentor’s observations for your child along with a 
    personalized study plan (attached to this email).

    Mentor's Comments:
    ---------------------------------------------------------
    {comments}
    ---------------------------------------------------------

    The attached study plan has been carefully designed to address your child’s 
    specific needs and ensure balanced progress across all subjects. 
    We encourage you to review it and provide a supportive environment 
    at home to help your child follow the schedule consistently.

    Should you have any questions or wish to discuss further, 
    please feel free to reach out to the school mentor team.

    Warm regards,  
    Lumina AI Team
    """

    msg.attach(MIMEText(body, "plain"))

    # Attach PDF if available
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="pdf")
            part.add_header("Content-Disposition", "attachment", filename=os.path.basename(pdf_path))
            msg.attach(part)

    # Send email
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())

    print("Email sent successfully!")



################################################################
#                       End Points                             #
################################################################

@app.get("/HealthCheck")
async def check():
    return{"Server running successfully"}


@app.post("/predict")
def predict(student: StudentInput):
    # --- Rule Based ---
    if (
        student.absence_days > 50 or
        student.physics_score < 20 or
        student.chemistry_score < 20 or
        student.math_score < 20 or
        student.weekly_self_study_hours < 2
    ):
        reasons = []
        if student.absence_days > 50: reasons.append("High Absence")
        if student.physics_score < 20: reasons.append("Low Physics Score")
        if student.chemistry_score < 20: reasons.append("Low Chemistry Score")
        if student.math_score < 20: reasons.append("Low Math Score")
        if student.weekly_self_study_hours < 2: reasons.append("Low Self Study Hours")

        risklevel = "High Risk"
        result = {"RISKLevel": risklevel, "Reasons": reasons}
    else:
        # --- ML Based ---
        X_new = pd.DataFrame([student.dict()])
        X_new = pd.get_dummies(X_new, drop_first=True)
        X_new = X_new.reindex(columns=model_columns, fill_value=0)

        debarred = clf_binary.predict(X_new)[0]

        y_prob = clf_multi.predict_proba(X_new)
        predicted_reasons = []
        for i, probs in enumerate(y_prob):
            if probs[0][1] > 0.3:  # threshold
                predicted_reasons.append(mlb.classes_[i])

        if debarred == 1 and len(predicted_reasons) == 0:
            predicted_reasons = ["Unknown Risk"]

        if debarred == 1:
            if len(predicted_reasons) == 1:
                risklevel = "Low Risk"
            elif len(predicted_reasons) <= 3:
                risklevel = "Medium Risk"
            else:
                risklevel = "High Risk"
        else:
            risklevel = "No Risk"

        result = {
            "RISKLevel": risklevel,
            "Reasons": predicted_reasons if debarred else ["None"]
        }

    # ✅ Update Firestore
    student_ref = db.collection("students").document(str(student.id))
    student_ref.set({
        "risk_level": result["RISKLevel"],
        "risk_reasons": result["Reasons"],
        "updated_at": firestore.SERVER_TIMESTAMP
    }, merge=True)

    return result


@app.post("/predict-all")
def predict_all():
    students = db.collection("students").stream()
    results = []

    for s in students:
        student_data = s.to_dict()
        try:
            student = StudentInput(**{
                "id": int(student_data.get("id")),
                "gender": student_data.get("gender", "Unknown"),
                "math_score": int(student_data.get("math_score", 0)),
                "physics_score": int(student_data.get("physics_score", 0)),
                "chemistry_score": int(student_data.get("chemistry_score", 0)),
                "path_time_job": student_data.get("path_time_job", "No"),
                "absence_days": int(student_data.get("absence_days", 0)),
                "extracurricular_activities": student_data.get("extracurricular_activities", "No"),
                "weekly_self_study_hours": int(student_data.get("weekly_self_study_hours", 0)),
                "career_aspiration": student_data.get("career_aspiration", "Unknown"),
                "fee_paid": int(student_data.get("fee_paid", 0)),
                "fee_pending": int(student_data.get("fee_pending", 0)),
            })

            result = predict(student)

            # ✅ Update Firestore for each student
            student_ref = db.collection("students").document(str(student.id))
            student_ref.set({
                "risk_level": result["RISKLevel"],
                "risk_reasons": result["Reasons"],
                "updated_at": firestore.SERVER_TIMESTAMP
            }, merge=True)

            results.append({"id": s.id, **result})

        except Exception as e:
            logger.error(f"Error processing student {s.id}: {e}")

    return {"results": results}


@app.post("/studyplan")
async def get_studyplan(req: StudyPlanRequest):
    if not req.comments:
        return {"error": "No comments provided"}

    study = StudyPlan()
    student_details = study.input_parser(req.comments)
    plan = result.invoke(student_details)
    generate_pdf_report(plan["StudyPlan"])
    send_email_gmail(
        req.email,
        req.comments,
        plan["StudyPlan"],
        pdf_path="study_plan_report.pdf"
    )
    return {"StudyPlan": plan}



