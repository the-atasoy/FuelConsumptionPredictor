import pickle
import pandas as pd

def load_model():
    with open('lasso_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def preprocess_input(data):
    data = pd.DataFrame(data, index=[0])

    feature_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin_america', 'origin_europa', 'origin_japan']
    data = data[feature_columns]

    _, scaler = load_model()
    data_scaled = pd.DataFrame(scaler.transform(data))
    return data_scaled