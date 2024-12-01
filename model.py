import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# Function to load the model and scaler
def load_model():
    with open('lasso_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Function to preprocess the input features
def preprocess_input(data):
    data = pd.DataFrame(data, index=[0])

    # Standardize features
    feature_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin_america', 'origin_europa', 'origin_japan']
    data = data[feature_columns]

    # Load the scaler and transform
    _, scaler = load_model()
    data_scaled = pd.DataFrame(scaler.transform(data))
    return data_scaled