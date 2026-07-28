import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE


def select_features_rf_importance(X_train, y_train, top_n=20):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    print(importances.head(top_n))
    return importances.head(top_n).index.tolist(), importances


def select_features_rfe(X_train, y_train, n_features=15):
    estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    selected = X_train.columns[rfe.support_].tolist()
    print(f"RFE selected features: {selected}")
    return selected


def save_final_feature_datasets(final_features, target='Appliances'):

    forced_features = ['hour_sin', 'hour_cos']
    all_features = sorted(set(final_features) | set(forced_features))
    target_cols = ['Appliances', 'Appliances_log']

    for suffix in ['unscaled', 'nn_scaled']:
        for split in ['train', 'test']:
            path = f'data/scaled/{split}_{suffix}.csv'
            df = pd.read_csv(path, index_col='date', parse_dates=True)

            keep_cols = [c for c in all_features + target_cols if c in df.columns]
            df_final = df[keep_cols]

            out_path = f'data/scaledfinal/{split}_{suffix}_final.csv'
            df_final.to_csv(out_path)
            print(f"Saved {out_path} — shape {df_final.shape}")


if __name__ == '__main__':
    import os

    os.makedirs('data/scaledfinal', exist_ok=True)

    train = pd.read_csv('data/scaled/train_unscaled.csv', index_col='date', parse_dates=True)
    drop_cols = ['Appliances', 'Appliances_log'] + \
                [c for c in train.columns if c.startswith('Appliances_log_lag') or c.startswith('Appliances_log_roll')]

    X_train = train.drop(columns=drop_cols)
    y_train = train['Appliances']

    top_rf, importances = select_features_rf_importance(X_train, y_train, top_n=20)
    top_rfe = select_features_rfe(X_train, y_train, n_features=15)

    # rv1/rv2 get dropped here
    final_features = sorted(set(top_rf) & set(top_rfe))
    print(f"\nFeatures selected by BOTH RF importance and RFE: {final_features}")
    print(f"rv1 rank: {importances.get('rv1', 'not in top 20')}")
    print(f"rv2 rank: {importances.get('rv2', 'not in top 20')}")

    pd.Series(final_features).to_csv('data/selected_features.csv', index=False)

    save_final_feature_datasets(final_features)