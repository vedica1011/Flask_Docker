from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd


print('pandas version',pd.__version__)
print('pickle version',pickle.format_version)

with open('linear_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])

def predict():
    if request.method == 'POST':
        try:
            MedInc = float(request.form['MedInc'])
            HouseAge = float(request.form['HouseAge'])
            AveRooms = float(request.form['AveRooms'])
            AveBedrms = float(request.form['AveBedrms'])
            Population = float(request.form['Population'])
            AveOccup = float(request.form['AveOccup'])
            Latitude = float(request.form['Latitude'])
            Longitude = float(request.form['Longitude'])

            pred_args = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
            input_val = pd.DataFrame([pred_args],
                                columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])

            pred_value = loaded_model.predict(input_val)[0]
        except ValueError:
            return "Please check input values..."
            
    return render_template('result.html', prediction=pred_value)


# Dummy route to handle favicon request
@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(port=5000, debug=True,host='0.0.0.0')
