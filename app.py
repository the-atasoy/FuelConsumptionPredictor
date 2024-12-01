from flask import Flask, request, render_template
import numpy as np
from model import load_model, preprocess_input

app = Flask(__name__)

model, scaler = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        cylinders = int(request.form['cylinders'])
        displacement = float(request.form['displacement'])
        horsepower = float(request.form['horsepower'])
        weight = float(request.form['weight'])
        acceleration = float(request.form['acceleration'])
        model_year = int(request.form['model_year'])
        origin = int(request.form['origin'])

        if (origin == 1):
            origin_america = 1
            origin_europa = 0
            origin_japan = 0
        elif (origin == 2):
            origin_europa = 1
            origin_america = 0
            origin_japan = 0
        else:
            origin_japan = 1
            origin_america = 0
            origin_europa = 0

        data = {
            'cylinders': cylinders,
            'displacement': displacement,
            'horsepower': horsepower,
            'weight': weight,
            'acceleration': acceleration,
            'model year': model_year,
            'origin_america': origin_america,
            'origin_europa': origin_europa,
            'origin_japan': origin_japan
        }
        
        data_scaled = preprocess_input(data)

        prediction = model.predict(data_scaled)
        return render_template('index.html', prediction_text=f'Predicted MPG: {prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)