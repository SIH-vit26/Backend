import pandas as pd
import random
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

fake = Faker("en_IN")

NUM_STUDENTS = 50
TOTAL_WORKING_DAYS = 200
TOTAL_FEE = 100000  

RISK_CATEGORIES = [
    "Low Attendance", "Low Average Marks",
    "Multiple Failures", "Low Study + Low Marks",
    "Fee Default", "High Risk Combo", "Admin Debar", "None"
]

def generate_student(force_risk="None", sid=0):
    first_name = fake.first_name()
    last_name = fake.last_name()
    email = f"{first_name.lower()}.{last_name.lower()}@example.com"
    gender = random.choice(["Male", "Female"])

    scores = {
        "math_score": random.randint(20, 100),
        "physics_score": random.randint(20, 100),
        "chemistry_score": random.randint(20, 100),
    }
    absence_days = random.randint(0, 80)
    extracurricular = random.choice(["Yes", "No"])
    fee_paid = random.randint(20000, TOTAL_FEE)
    fee_pending = TOTAL_FEE - fee_paid
    fee_due_date = fake.date_between(start_date="-6M", end_date="+3M")
    path_time_job = random.choice(["Yes", "No"])
    weekly_self_study_hours = random.randint(0, 30)
    career_aspiration = random.choice(
        ["Doctor", "Engineer", "Artist", "Scientist", "Teacher", "Entrepreneur"]
    )

    if force_risk == "Low Attendance":
        absence_days = random.randint(45, 80)
    if force_risk == "Low Average Marks":
        scores = {k: random.randint(20, 34) for k in scores}
    if force_risk == "Multiple Failures":
        scores = {"math_score": random.randint(20, 30),
                  "physics_score": random.randint(20, 30),
                  "chemistry_score": random.randint(20, 100)}
    if force_risk == "Low Study + Low Marks":
        weekly_self_study_hours = random.randint(0, 3)
        scores = {k: random.randint(20, 40) for k in scores}
    if force_risk == "Fee Default":
        fee_paid = random.randint(20000, 40000)
        fee_pending = TOTAL_FEE - fee_paid
        fee_due_date = fake.date_between(start_date="-6M", end_date="today") - timedelta(days=1)
    if force_risk == "High Risk Combo":
        absence_days = random.randint(50, 80)
        scores = {k: random.randint(20, 34) for k in scores} 
    if force_risk == "Admin Debar":
        absence_days = random.randint(60, 80)                
        fee_paid = random.randint(20000, 40000)              
        fee_pending = TOTAL_FEE - fee_paid
        fee_due_date = fake.date_between(start_date="-6M", end_date="today") - timedelta(days=1)

 
    avg_score = sum(scores.values()) / len(scores)
    failed_subjects = sum([1 for v in scores.values() if v < 33])

    risks = []
    if absence_days > 0.2 * TOTAL_WORKING_DAYS:
        risks.append("Low Attendance")
    if avg_score < 35:
        risks.append("Low Average Marks")
    if failed_subjects >= 2:
        risks.append("Multiple Failures")
    if weekly_self_study_hours < 5 and avg_score < 40:
        risks.append("Low Study + Low Marks")
    if fee_pending > 0.5 * TOTAL_FEE and fee_due_date < datetime.today().date():
        risks.append("Fee Default")
    if ("Low Attendance" in risks and "Low Average Marks" in risks):
        risks.append("High Risk Combo")
    if ("Fee Default" in risks and "Low Attendance" in risks):
        risks.append("Admin Debar")

    return {
        "id": sid,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "gender": gender,
        **scores,
        "path_time_job": path_time_job,
        "absence_days": absence_days,
        "extracurricular_activities": extracurricular,
        "weekly_self_study_hours": weekly_self_study_hours,
        "career_aspiration": career_aspiration,
        "fee_paid": fee_paid,
        "fee_pending": fee_pending,
        "fee_due_date": fee_due_date,
        "debar_risk": ", ".join(risks) if risks else "None"
    }

students = []
per_class = NUM_STUDENTS // len(RISK_CATEGORIES)
sid = 1
for risk in RISK_CATEGORIES:
    for _ in range(per_class):
        students.append(generate_student(force_risk=risk, sid=sid))
        sid += 1

df = pd.DataFrame(students)


def add_noise(series, std, min_val, max_val):
    noisy = series + np.random.normal(0, std, len(series))
    return np.clip(noisy, min_val, max_val).astype(int)

for subj in ["math_score", "physics_score", "chemistry_score"]:
    df[subj] = add_noise(df[subj], std=5, min_val=0, max_val=100)

df["absence_days"] = add_noise(df["absence_days"], std=2, min_val=0, max_val=80)
df["weekly_self_study_hours"] = add_noise(df["weekly_self_study_hours"], std=1, min_val=0, max_val=30)

for col in ["gender", "path_time_job", "extracurricular_activities"]:
    mask = np.random.rand(len(df)) < 0.03
    if col == "gender":
        df.loc[mask, col] = df.loc[mask, col].map({"Male": "Female", "Female": "Male"})
    else:
        df.loc[mask, col] = df.loc[mask, col].map({"Yes": "No", "No": "Yes"})

df["fee_due_date"] = pd.to_datetime(df["fee_due_date"]) + pd.to_timedelta(
    np.random.randint(-5, 5, len(df)), unit="D"
)


attendance_df = df[["id", "first_name", "last_name", "absence_days", "extracurricular_activities", "debar_risk"]]
marks_df = df[["id", "first_name", "last_name", "math_score", "physics_score", "chemistry_score", "debar_risk"]]
fee_df = df[["id", "first_name", "last_name", "email", "fee_paid", "fee_pending", "fee_due_date", "debar_risk"]]

attendance_df.to_csv("attendance_db.csv", index=False)
marks_df.to_csv("marks_db.csv", index=False)
fee_df.to_csv("fee_db.csv", index=False)
df.to_csv("student_master_balanced.csv", index=False)

print("✅ Balanced dataset generated with all risk factors included for combos.")
