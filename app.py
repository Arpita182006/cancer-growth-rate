from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import zipfile

app = Flask(__name__)

FEATURES = ["Age","Gender","Previous_Tumor_Size","Time_Since_Diagnosis","Stage","Genetic_Risk","Smoking_History","Alcohol_Consumption","BMI","Treatment_Chemotherapy","Treatment_Radiotherapy","Treatment_Surgery"]
model = None
model_error = None

def load_model():
    global model, model_error
    try:
        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                model = pickle.load(f)
            return
        if os.path.exists("model.zip"):
            with zipfile.ZipFile("model.zip") as z:
                files = [n for n in z.namelist() if n.lower().endswith((".pkl",".pickle",".joblib"))]
                if not files:
                    raise FileNotFoundError("No model file found inside model.zip.")
                name = files[0]
                with z.open(name) as src, open(os.path.basename(name), "wb") as dst:
                    dst.write(src.read())
                with open(os.path.basename(name), "rb") as f:
                    model = pickle.load(f)
            return
        raise FileNotFoundError("model.pkl or model.zip was not found.")
    except Exception as e:
        model_error = str(e)

load_model()

@app.route("/", methods=["GET","POST"])
def home():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            if model is None:
                raise RuntimeError(model_error or "Model could not be loaded.")
            values = [[
                int(request.form["age"]), int(request.form["gender"]),
                float(request.form["previous_tumor_size"]), int(request.form["time_since_diagnosis"]),
                int(request.form["stage"]), int(request.form["genetic_risk"]),
                int(request.form["smoking"]), int(request.form["alcohol"]),
                float(request.form["bmi"]), int(request.form.get("treatment_chemo",0)),
                int(request.form.get("treatment_radio",0)), int(request.form.get("treatment_surgery",0))
            ]]
            prediction = round(float(model.predict(pd.DataFrame(values, columns=FEATURES))[0]), 2)
        except Exception as e:
            error = str(e)
    return render_template("index.html", prediction=prediction, error=error, model_error=model_error)

if __name__ == "__main__":
    app.run(debug=True)
