import pandas as pd

attendance_df = pd.read_csv("attendance_db.csv")
marks_df = pd.read_csv("marks_db.csv")
fee_df = pd.read_csv("fee_db.csv")

for df in [attendance_df, marks_df, fee_df]:
    df["id"] = df["id"].astype(int)

merged_df = marks_df.merge(attendance_df, on=["id", "first_name", "last_name"], how="outer")
merged_df = merged_df.merge(fee_df, on=["id", "first_name", "last_name"], how="outer")

if "debar_risk_x" in merged_df.columns and "debar_risk_y" in merged_df.columns:
    merged_df["debar_risk"] = merged_df["debar_risk_x"].combine_first(merged_df["debar_risk_y"])
    merged_df = merged_df.drop(columns=["debar_risk_x", "debar_risk_y"])

merged_df.to_csv("student_master.csv", index=False)

print("✅ Merged dataset saved as student_master.csv")
print(merged_df.head())
