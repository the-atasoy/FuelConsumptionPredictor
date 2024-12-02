import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

df = pd.read_csv('data/auto-mpg.csv')

def __preprocess():
    df.drop(['car name'], axis=1, inplace=True)
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors='coerce')
    df.fillna({'horsepower': df['horsepower'].median()}, inplace=True)
    df['model year'] = df['model year'].apply(lambda x: 1900 + x)


def __one_hot_encode():
    global df
    df_one_hot = pd.get_dummies(df['origin'], prefix='', prefix_sep='')
    df_one_hot.columns = ['origin_america', 'origin_europa', 'origin_japan']
    df = pd.concat([df.drop(columns=['origin']), df_one_hot], axis=1)
    df[['origin_america', 'origin_europa', 'origin_japan']] = df[['origin_america', 'origin_europa', 'origin_japan']].astype(int)

def __handle_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column_name] = df[column_name].apply(
        lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
    )

    return df

def __split_dataset():
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

def __scale_features(test_dataset, train_dataset):
    scaler_standard = StandardScaler()
    train_dataset = scaler_standard.fit_transform(train_dataset)
    test_dataset = scaler_standard.transform(test_dataset)
    return test_dataset, train_dataset, scaler_standard

def __train_model(x_train, y_train):
    lasso_model = Lasso(alpha=0.01)
    return lasso_model.fit(x_train, y_train)

def main():
    __preprocess()
    __one_hot_encode()
    __handle_outliers(df, 'horsepower')
    __handle_outliers(df, 'acceleration')
    x_train, x_test, y_train, y_test = __split_dataset()
    _, x_train_scaled, scaler = __scale_features(x_test, x_train)
    model = __train_model(x_train_scaled, y_train)
    with open('lasso_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == '__main__':
    main()