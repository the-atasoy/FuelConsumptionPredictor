from flask import Flask, request, render_template
import numpy as np
from model import load_model, preprocess_input

app = Flask(__name__)

# Load the model
model, scaler = load_model()

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        cylinders = int(request.form['cylinders'])
        displacement = float(request.form['displacement'])
        horsepower = float(request.form['horsepower'])
        weight = float(request.form['weight'])
        acceleration = float(request.form['acceleration'])
        model_year = int(request.form['model_year'])
        origin = int(request.form['origin'])  # 1 for USA, 2 for Europe, 3 for Japan

        # Prepare the data in the format required by the model
        data = {
            'cylinders': cylinders,
            'displacement': displacement,
            'horsepower': horsepower,
            'weight': weight,
            'acceleration': acceleration,
            'model year': model_year,
            'origin': origin
        }
        
        data_scaled = preprocess_input(data)

        # Make the prediction
        prediction = model.predict(data_scaled)
        return render_template('index.html', prediction_text=f'Predicted MPG: {prediction[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)