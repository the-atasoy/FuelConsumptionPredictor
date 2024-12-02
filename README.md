# Fuel Consumption Predictor
cikcik
This project predicts fuel consumption (in liters per 100km) based on various car attributes using a Lasso regression model.

## Project Structure

- `app.py`: Flask application to serve the prediction model.
- `data/`: Directory containing the dataset and serialized model/scaler.
- `model.py`: Contains functions to load the model and preprocess input data.
- `playground/`: Directory containing Jupyter notebooks for experiments on data.
- `requirements.txt`: List of dependencies.
- `train_model.py`: Script to preprocess data, train the model, and save the model/scaler.

## Setup

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and activate it (project has been coded in Python 3.12.3):
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

1. Run the `train_model.py` script to preprocess the data, train the model, and save the model and scaler:
    ```sh
    python train_model.py
    ```

### Running the Flask App

1. Start the Flask application:
    ```sh
    python app.py
    ```

2. The application will be available at `http://127.0.0.1:5000`.

### Making Predictions

1. Send a POST request to `http://127.0.0.1:5000/predict` with a JSON payload containing the car attributes:
    ```json
    {
        "cylinders": 4,         // Number of cylinders
        "displacement": 2,      // Engine displacement in liters
        "horsepower": 88,       // Engine horsepower
        "weight": 1440,         // Vehicle weight in kilogram
        "acceleration": 15.4,   // Time to accelerate from 0 to 100 kmh in seconds
        "model_year": 1978,     // Model year of the car
        "origin": 1             // Origin of the car (1: America, 2: Europe, 3: Japan)
    }
    ```

2. Response:
    ```json
    {
        "prediction": 7.69      // Predicted fuel consumption in liters per 100km
    }
    ```

## Data

The dataset used for training the model is `data/auto-mpg.csv`, which contains the following columns:
- `mpg`: Miles per gallon
- `cylinders`: Number of cylinders
- `displacement`: Engine displacement in cubic inch
- `horsepower`: Engine horsepower
- `weight`: Vehicle weight in pound
- `acceleration`: Time to accelerate from 0 to 60 mph
- `model year`: Model year of the car as last 2 digits
- `origin`: Origin of the car (1: America, 2: Europe, 3: Japan)
- `car name`: Name of the car
