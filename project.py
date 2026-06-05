import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("student_dataset.csv")

# ---------------------------------------------------
# ENCODING
# ---------------------------------------------------

le_consistency = LabelEncoder()
le_env = LabelEncoder()
le_support = LabelEncoder()
le_coaching = LabelEncoder()
le_result = LabelEncoder()

df["StudyConsistency"] = le_consistency.fit_transform(df["StudyConsistency"])
df["StudyEnvironment"] = le_env.fit_transform(df["StudyEnvironment"])
df["FamilySupport"] = le_support.fit_transform(df["FamilySupport"])
df["ExtraCoaching"] = le_coaching.fit_transform(df["ExtraCoaching"])
df["Result"] = le_result.fit_transform(df["Result"])

# ---------------------------------------------------
# FEATURES & TARGET
# ---------------------------------------------------

X = df.drop("Result", axis=1)
y = df["Result"]

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# ---------------------------------------------------
# GUI WINDOW
# ---------------------------------------------------

root = tk.Tk()
root.title("Student Performance Predictor")
root.geometry("500x650")

entries = {}

fields = [
    "StudyHours",
    "Attendance",
    "PreviousScore",
    "AssignmentMarks",
    "QuizMarks",
    "MidtermMarks",
    "SleepHours",
    "SocialMediaUsage",
    "InternetUsage"
]

# INPUT FIELDS
row = 0
for field in fields:
    tk.Label(root, text=field).grid(row=row, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=row, column=1)
    entries[field] = entry
    row += 1

# DROPDOWNS
def add_dropdown(label, options):
    global row
    tk.Label(root, text=label).grid(row=row, column=0)
    var = tk.StringVar()
    var.set(options[0])
    dropdown = tk.OptionMenu(root, var, *options)
    dropdown.grid(row=row, column=1)
    entries[label] = var
    row += 1

add_dropdown("StudyConsistency", ["Low", "Medium", "High"])
add_dropdown("StudyEnvironment", ["Poor", "Average", "Good"])
add_dropdown("FamilySupport", ["Low", "Medium", "High"])
add_dropdown("ExtraCoaching", ["No", "Yes"])

# ---------------------------------------------------
# PREDICTION FUNCTION (IMPROVED)
# ---------------------------------------------------

def predict():
    try:
        input_data = [
            float(entries["StudyHours"].get()),
            float(entries["Attendance"].get()),
            float(entries["PreviousScore"].get()),
            float(entries["AssignmentMarks"].get()),
            float(entries["QuizMarks"].get()),
            float(entries["MidtermMarks"].get()),
            float(entries["SleepHours"].get()),
            float(entries["SocialMediaUsage"].get()),
            float(entries["InternetUsage"].get()),

            le_consistency.transform([entries["StudyConsistency"].get()])[0],
            le_env.transform([entries["StudyEnvironment"].get()])[0],
            le_support.transform([entries["FamilySupport"].get()])[0],
            le_coaching.transform([entries["ExtraCoaching"].get()])[0],
        ]

        prediction = model.predict([input_data])[0]
        prob = model.predict_proba([input_data])[0][1]
        percentage = round(prob * 100, 2)

        result = le_result.inverse_transform([prediction])[0]

        # RISK LEVEL
        if prob < 0.4:
            risk = "High Risk"
        elif prob < 0.7:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        # RECOMMENDATION SYSTEM ⭐
        if risk == "High Risk":
            suggestion = "Increase study hours + reduce social media usage"
        elif risk == "Medium Risk":
            suggestion = "Improve attendance + focus on weak subjects"
        else:
            suggestion = "Keep it up! Maintain consistency"

        messagebox.showinfo(
            "Prediction Result",
            f"Result: {result}\n"
            f"Risk Level: {risk}\n"
            f"Probability: {percentage}%\n"
            f"Suggestion: {suggestion}"
        )

    except Exception as e:
        messagebox.showerror("Error", f"Check input values!\n{str(e)}")

# ---------------------------------------------------
# BUTTON
# ---------------------------------------------------

tk.Button(
    root,
    text="Predict",
    command=predict,
    bg="green",
    fg="white",
    font=("Arial", 12, "bold")
).grid(row=row+1, column=0, columnspan=2, pady=20)

root.mainloop()