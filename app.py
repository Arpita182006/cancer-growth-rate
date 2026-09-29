from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"
model = None
model_error = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
    else:
        model_error = "model.pkl was not found. Add your trained model.pkl to the project folder."
except Exception as e:
    model_error = f"Could not load model.pkl: {e}"

FEATURES = [
    "Age",
    "Gender",
    "Previous_Tumor_Size",
    "Time_Since_Diagnosis",
    "Stage",
    "Genetic_Risk",
    "Smoking_History",
    "Alcohol_Consumption",
    "BMI",
    "Treatment_Chemotherapy",
    "Treatment_Radiotherapy",
    "Treatment_Surgery"
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            if model is None:
                raise RuntimeError(model_error or "Model could not be loaded.")

            gender = int(request.form["gender"])
            stage = int(request.form["stage"])
            genetic_risk = int(request.form["genetic_risk"])
            smoking = int(request.form["smoking"])
            alcohol = int(request.form["alcohol"])

            treatment_chemo = int(request.form.get("treatment_chemo", 0))
            treatment_radio = int(request.form.get("treatment_radio", 0))
            treatment_surgery = int(request.form.get("treatment_surgery", 0))

            values = [[
                int(request.form["age"]),
                gender,
                float(request.form["previous_tumor_size"]),
                int(request.form["time_since_diagnosis"]),
                stage,
                genetic_risk,
                smoking,
                alcohol,
                float(request.form["bmi"]),
                treatment_chemo,
                treatment_radio,
                treatment_surgery
            ]]

            input_data = pd.DataFrame(values, columns=FEATURES)
            result = model.predict(input_data)[0]
            prediction = round(float(result), 2)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        model_error=model_error
    )

if __name__ == "__main__":
    app.run(debug=True)
