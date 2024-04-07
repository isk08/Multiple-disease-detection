from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from d_prediction import PredictionPipeline

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST': 
        try:
            pregnancies = float(request.form['pregnancies'])
            glucose = float(request.form['glucose'])
            bp = float(request.form['bp'])
            st = float(request.form['st'])
            insulin = float(request.form['insulin'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = float(request.form['age'])
            

            data = [pregnancies, glucose, bp, st, insulin, bmi, age]
            data = np.array(data).reshape(1, 8)

            obj = PredictionPipeline()
            predict = obj.predict(data)

            if predict == 1:
                predicted_text = "Heart Disease Predicted"
            else:
                predicted_text = "Heart Disease Not Predicted"

            return render_template('index1.html', predict=predicted_text)

        except Exception as e:
            return 'Something is wrong: ${e}'
    else:
        return render_template('index1.html')
    

if __name__ == "__main__":
    app.run(debug=True)
