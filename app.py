import pickle
import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route Home Page "templates/index.html" directory
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if(request.method == "GET"):
        return render_template("home.html")
    elif(request.method == "POST"):
        data = CustomData(
                    gender=request.form.get("gender"),
                    race_ethnicity=request.form.get("ethnicity"),
                    parental_level_of_education=request.form.get("parental_level_of_education"),
                    lunch=request.form.get("lunch"),
                    test_preparation_course=request.form.get("test_preparation_course"),
                    reading_score=float(request.form.get("reading_score")),
                    writing_score=float(request.form.get("writing_score"))
                )
        prediction_data_df = data.get_data_as_data_frame()

        print(prediction_data_df)

        predict_pipeline = PredictPipeline()
        prediction_result = predict_pipeline.predict(prediction_data_df)

        return render_template("home.html", results=prediction_result[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)