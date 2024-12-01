import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

df = pd.read_csv('data/auto-mpg.csv')

def preprocess():
    df.drop(['car name'], axis=1, inplace=True)
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors='coerce')
    df.fillna({'horsepower': df['horsepower'].median()}, inplace=True)

def one_hot_encode():
    global df
    df_one_hot = pd.get_dummies(df['origin'], prefix='', prefix_sep='')
    df_one_hot.columns = ['origin_america', 'origin_europa', 'origin_japan']
    df = pd.concat([df.drop(columns=['origin']), df_one_hot], axis=1)
    df[['origin_america', 'origin_europa', 'origin_japan']] = df[['origin_america', 'origin_europa', 'origin_japan']].astype(int)

def split_dataset():
    x = df[['cylinders',
            'displacement',
            'horsepower',
            'weight',
            'acceleration',
            'model year',
            'origin_america',
            'origin_europa',
            'origin_japan']]
    y = df['mpg']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def scale_features(test_dataset, train_dataset):
    scaler_standard = StandardScaler()
    test_dataset = scaler_standard.fit_transform(test_dataset)
    train_dataset = scaler_standard.transform(train_dataset)

def train_model(x_train, y_train):
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(x_train, y_train)
    return lasso_model.predict(x_train)

def main():
    preprocess()
    one_hot_encode()
    x_train, x_test, y_train, y_test = split_dataset()
    scale_features(x_test, x_train)
    return train_model(x_train, y_train)

if __name__ == '__main__':
    main()