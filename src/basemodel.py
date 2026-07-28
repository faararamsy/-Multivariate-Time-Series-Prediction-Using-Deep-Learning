"""
Baseline models: naive persistence, Linear Regression, Random Forest, XGBoost.
Evaluated with walk-forward (TimeSeriesSplit) validation.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error — guard against divide-by-zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred, model_name=''):
    """Compute the full metric set for one model's predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mp = mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mp, 'R2': r2}


def naive_persistence_baseline(df, target='Appliances'):
    """
    Predicts Appliances[t] = Appliances[t-1].
    CRITICAL baseline: feature importance showed Appliances_lag_1 alone
    explains ~60% of RF importance. Every real model MUST beat this,
    or the extra complexity isn't earning its keep.
    """
    y_true = df[target].iloc[1:]
    y_pred = df[target].shift(1).iloc[1:]
    return evaluate(y_true, y_pred, model_name='Naive Persistence (t-1)')


def walk_forward_validate(model, X, y, n_splits=5):
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)

        metrics = evaluate(y_val_fold, preds, model_name=f'Fold {fold+1}')
        fold_results.append(metrics)

    df_results = pd.DataFrame(fold_results)
    summary = {
        'MAE_mean': df_results['MAE'].mean(), 'MAE_std': df_results['MAE'].std(),
        'RMSE_mean': df_results['RMSE'].mean(), 'RMSE_std': df_results['RMSE'].std(),
        'MAPE_mean': df_results['MAPE'].mean(), 'MAPE_std': df_results['MAPE'].std(),
        'R2_mean': df_results['R2'].mean(), 'R2_std': df_results['R2'].std(),
    }
    return df_results, summary


def diagnostic_check(train, test, target='Appliances'):
    """Run diagnostics to understand why RF/XGB are failing."""
    print("\n" + "="*60)
    print("DIAGNOSTIC CHECK")
    print("="*60)

    # Target distribution
    print("\n=== TARGET DISTRIBUTION CHECK ===")
    print(f"Train target mean: {train[target].mean():.2f}, std: {train[target].std():.2f}")
    print(f"Test target mean: {test[target].mean():.2f}, std: {test[target].std():.2f}")
    print(f"Train target 25th: {train[target].quantile(0.25):.2f}, 75th: {train[target].quantile(0.75):.2f}")
    print(f"Test target 25th: {test[target].quantile(0.25):.2f}, 75th: {test[target].quantile(0.75):.2f}")

    # Lag correlation
    if 'Appliances_lag_1' in train.columns:
        print("\n=== LAG_1 PREDICTION CHECK ===")
        train_lag_corr = train['Appliances_lag_1'].corr(train[target])
        test_lag_corr = test['Appliances_lag_1'].corr(test[target])
        print(f"Train: Correlation between Appliances_lag_1 and target: {train_lag_corr:.4f}")
        print(f"Test: Correlation between Appliances_lag_1 and target: {test_lag_corr:.4f}")

        # Naive MAE
        train_mae_naive = mean_absolute_error(train[target].iloc[1:], train[target].shift(1).iloc[1:])
        test_mae_naive = mean_absolute_error(test[target].iloc[1:], test[target].shift(1).iloc[1:])
        print(f"Train Naive Persistence MAE: {train_mae_naive:.4f}")
        print(f"Test Naive Persistence MAE: {test_mae_naive:.4f}")

    # Feature check
    print("\n=== FEATURE CHECK ===")
    drop_cols = ['Appliances', 'Appliances_log']
    feature_cols = [c for c in train.columns if c not in drop_cols]
    print(f"Number of features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols[:10]}...")

    if 'Appliances_lag_1' in feature_cols:
        print(f"✓ 'Appliances_lag_1' is in feature set")
        print(f"  Train NaN count: {train['Appliances_lag_1'].isna().sum()}")
        print(f"  Test NaN count: {test['Appliances_lag_1'].isna().sum()}")
    else:
        print("✗ WARNING: 'Appliances_lag_1' is NOT in feature set!")

    print("="*60 + "\n")

    return feature_cols


def train_baselines(train_path, test_path, target='Appliances'):

    print("\n" + "="*60)
    print("BASELINE MODELS TRAINING")
    print("="*60)

    # Load data
    train = pd.read_csv(train_path, index_col='date', parse_dates=True)
    test = pd.read_csv(test_path, index_col='date', parse_dates=True)

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Run diagnostics
    feature_cols = diagnostic_check(train, test, target)

    # Prepare data
    X_train, y_train = train[feature_cols], train[target]
    X_test, y_test = test[feature_cols], test[target]

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    results = []

    # Naive Persistence Baseline ---
    print("\n--- Training Naive Persistence ---")
    full_unscaled = pd.concat([train, test])
    naive_result = naive_persistence_baseline(full_unscaled.loc[test.index[0]:], target=target)
    results.append(naive_result)
    print(f"Naive Persistence MAE: {naive_result['MAE']:.4f}, R2: {naive_result['R2']:.4f}")

    #  Decision Tree ---
    print("\n--- Training Decision Tree (lag_1 only) ---")
    dt_simple = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt_simple.fit(X_train[['Appliances_lag_1']], y_train)
    dt_preds = dt_simple.predict(X_test[['Appliances_lag_1']])
    dt_result = evaluate(y_test, dt_preds, model_name='Decision Tree (lag_1 only)')
    results.append(dt_result)
    print(f"Decision Tree MAE: {dt_result['MAE']:.4f}, R2: {dt_result['R2']:.4f}")

    # --- 3. Random Forest (Optimized) ---
    print("\n--- Training Random Forest ---")
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_result = evaluate(y_test, rf_preds, model_name='Random Forest')
    results.append(rf_result)
    print(f"Random Forest MAE: {rf_result['MAE']:.4f}, R2: {rf_result['R2']:.4f}")

    # Feature importance
    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nTop 5 RF features:\n{importance.head(5)}")

    # --- 4. XGBoost ---
    print("\n--- Training XGBoost ---")
    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,           # Shallow trees prevent overfitting
        min_child_weight=3,    # Prevents overfitting
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.1,             # Minimum loss reduction for split
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=1.0,        # L2 regularization
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_result = evaluate(y_test, xgb_preds, model_name='XGBoost')
    results.append(xgb_result)
    print(f"XGBoost MAE: {xgb_result['MAE']:.4f}, R2: {xgb_result['R2']:.4f}")

    # Linear Regression (Log Target) ---
   
    print("\n--- Training Linear Regression (log target) ---")
    lr_results, lr_model = train_linear_regression(train_path, test_path)
    results.append(lr_results.iloc[0].to_dict())
    print(f"Linear Regression MAE: {lr_results.iloc[0]['MAE']:.4f}, R2: {lr_results.iloc[0]['R2']:.4f}")

    # Linear Regression 
    print("\n--- Training Linear Regression (standard) ---")
    lr_std = LinearRegression()
    lr_std.fit(X_train, y_train)
    lr_std_preds = lr_std.predict(X_test)
    lr_std_result = evaluate(y_test, lr_std_preds, model_name='Linear Regression (standard)')
    results.append(lr_std_result)
    print(f"Linear Regression (standard) MAE: {lr_std_result['MAE']:.4f}, R2: {lr_std_result['R2']:.4f}")

    # --- Summary ---
    df_results = pd.DataFrame(results)
    print("\n" + "="*60)
    print("BASELINE RESULTS SUMMARY")
    print("="*60)
    print(df_results[['model', 'MAE', 'RMSE', 'MAPE', 'R2']].to_string(index=False))

    # Save results
    df_results.to_csv('reports/baseline_results.csv', index=False)
    print("\nResults saved to 'reports/baseline_results.csv'")

    # Save models
    import joblib
    joblib.dump(rf, 'models/random_forest.pkl')
    joblib.dump(xgb, 'models/xgboost.pkl')
    joblib.dump(lr_std, 'models/linear_regression.pkl')
    print("Models saved to 'models/' directory")

    return df_results, rf, xgb, lr_std


def train_linear_regression(train_path, test_path, target='Appliances_log'):

    train = pd.read_csv(train_path, index_col='date', parse_dates=True)
    test = pd.read_csv(test_path, index_col='date', parse_dates=True)

    drop_cols = ['Appliances', 'Appliances_log']
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X_train, y_train = train[feature_cols], train[target]
    X_test = test[feature_cols]
    y_test_real = test['Appliances']  # compare against REAL Wh, not log values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    preds_log = lr.predict(X_test)
    preds_real = np.expm1(preds_log)  # invert log1p to get back to real Wh

    result = evaluate(y_test_real, preds_real, model_name='Linear Regression (log target)')
    return pd.DataFrame([result]), lr


if __name__ == '__main__':
    import os

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('report', exist_ok=True)

    print("\n" + "="*60)
    print("BASELINE MODELS - FULL PIPELINE")
    print("="*60)

    # All baseline models (including Linear Regression) now read from the
    # SAME unscaled final feature set. 
    tree_results, rf_model, xgb_model, lr_model = train_baselines(
        'data/scaledfinal/train_unscaled_final.csv',
        'data/scaledfinal/test_unscaled_final.csv',
        target='Appliances'
    )

    print("\n" + "="*60)
    print("BASELINE MODELS COMPLETE!")
    print("="*60)