import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from src.feature_engineering import add_lagged_features, add_rolling_features


# adding a log transformed target for skew sensitive models
def add_logtrans_features(df, target='Appliances'):
    df[f'{target}_log'] = np.log1p(df[target])
    df = add_lagged_features(df, target=f'{target}_log', lags=(1, 2, 3, 6, 18, 144))
    df = add_rolling_features(df, target=f'{target}_log', windows=(6, 18))
    return df


def chronological_split(df, train_ratio=0.8):
    split_index = int(len(df) * train_ratio)
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    print(f"Train: {train.index.min()} to {train.index.max()} ({len(train)} rows)")
    print(f"Test:  {test.index.min()} to {test.index.max()} ({len(test)} rows)")
    return train, test


def minmaxscaler_nn(train, test, feature_cols):
    scaler = MinMaxScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[feature_cols] = scaler.fit_transform(train[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])
    return train_scaled, test_scaled, scaler


if __name__ == '__main__':
    import os

    os.makedirs('data/scaled', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    df = pd.read_csv('data/engineeredfeatures.csv', index_col='date', parse_dates=True)

    df = add_logtrans_features(df)
    train, test = chronological_split(df, train_ratio=0.8)

    # separate feature names from target column
    feature_cols = [c for c in df.columns if c not in ['Appliances', 'Appliances_log']]

    # unscaled version — used by tree models (RF, XGBoost, Decision Tree)
    # AND by Linear Regression, since OLS doesn't need scaling (see note above)
    train.to_csv('data/scaled/train_unscaled.csv')
    test.to_csv('data/scaled/test_unscaled.csv')

    # MinMax-scaled version — used by the NN/LSTM only
    train_nn, test_nn, minmax_scaler = minmaxscaler_nn(train, test, feature_cols)
    train_nn.to_csv('data/scaled/train_nn_scaled.csv')
    test_nn.to_csv('data/scaled/test_nn_scaled.csv')

    joblib.dump(minmax_scaler, 'models/minmax_scaler.pkl')

    print("\nSaved: train_unscaled.csv, test_unscaled.csv, train_nn_scaled.csv, test_nn_scaled.csv")
    print("Saved: models/minmax_scaler.pkl")