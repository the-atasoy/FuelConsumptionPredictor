from flask import Flask, request, jsonify
from model import load_model, preprocess_input

app = Flask(__name__)

model, scaler = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = __convert_request_to_model_input(request.get_json())
        
        data_scaled = preprocess_input(data)
        prediction = 235.215 / model.predict(data_scaled)[0] # convert mpg to liters per 100km

        return jsonify(
            {
            'prediction': round(prediction, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
def __convert_request_to_model_input(data):
    cylinders = int(data['cylinders'])
    displacement = float(data['displacement']) / 0.016387 # convert liters to cubic inches
    horsepower = float(data['horsepower'])
    weight = float(data['weight']) / 0.453592 # convert kg to lbs
    acceleration = float(data['acceleration'])
    model_year = int(data['model_year'])
    origin = int(data['origin'])

    origin_america = 1 if origin == 1 else 0
    origin_europa = 1 if origin == 2 else 0
    origin_japan = 1 if origin == 3 else 0

    input_data = {
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
    return input_data

if __name__ == '__main__':
    app.run(debug=True)