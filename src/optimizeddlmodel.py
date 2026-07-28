"""
Hyperparameter tuning via Optuna (Bayesian Optimization), applied to
ALL FOUR deep learning architectures — not just the winner — so every
model gets a fair before/after comparison, per PDF section 5.4.
"""

import os
import json
import warnings
import joblib
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Silence non-critical warnings and Optuna per-trial logging spam
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def build_model_with_params(model_type, input_shape, units_1, units_2, dropout_rate, learning_rate):
    """
    Rebuilds each architecture using Optuna-suggested hyperparameters.
    Note: recurrent_dropout is removed to allow GPU/cuDNN acceleration.
    """
    if model_type == 'LSTM':
        model = Sequential([
            LSTM(units_1, return_sequences=True, dropout=dropout_rate, input_shape=input_shape),
            LSTM(units_2, dropout=dropout_rate),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='linear')
        ])
    elif model_type == 'GRU':
        model = Sequential([
            GRU(units_1, return_sequences=True, dropout=dropout_rate, input_shape=input_shape),
            GRU(units_2, dropout=dropout_rate),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='linear')
        ])
    elif model_type == 'CNN-LSTM_v2':
        model = Sequential([
            Conv1D(filters=32, kernel_size=2, activation='relu', padding='same', input_shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            LSTM(units_1, return_sequences=True, dropout=dropout_rate),
            LSTM(units_2, dropout=dropout_rate),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='linear')
        ])
    elif model_type == 'Bi-LSTM':
        model = Sequential([
            Bidirectional(LSTM(units_1, return_sequences=True, dropout=dropout_rate), input_shape=input_shape),
            Bidirectional(LSTM(units_2, dropout=dropout_rate)),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='linear')
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber(delta=1.0), metrics=['mae'])
    return model


def objective(trial, model_type, X_tr, y_tr, X_val, y_val, input_shape):
    """
    One Optuna trial using Bayesian/TPE sampling to identify best hyperparameters.
    """
    units_1 = trial.suggest_int('units_1', 32, 96, step=16)
    units_2 = trial.suggest_int('units_2', 16, 64, step=16)
    dropout_rate = trial.suggest_float('dropout', 0.1, 0.4)
    learning_rate = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    model = build_model_with_params(model_type, input_shape, units_1, units_2, dropout_rate, learning_rate)

    early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    history = model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val),
        epochs=30, batch_size=batch_size, callbacks=[early_stop], verbose=0
    )

    return min(history.history['val_loss'])


def tune_model(model_type, X_tr, y_tr, X_val, y_val, input_shape, n_trials=10):
    """Runs Optuna hyperparameter optimization search for ONE architecture."""
    print(f"\n{'='*60}")
    print(f"TUNING {model_type} ({n_trials} trials)")
    print(f"{'='*60}")

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, model_type, X_tr, y_tr, X_val, y_val, input_shape),
        n_trials=n_trials
    )

    print(f"Best params for {model_type}: {study.best_params}")
    print(f"Best val_loss: {study.best_value:.5f}")
    return study.best_params


def train_final_and_evaluate(model_type, best_params, X_train, y_train, X_test, y_test,
                             input_shape, target_scaler, epochs=100):
    """
    Rebuilds model using best params, trains fully, evaluates performance metrics, 
    and returns both the metrics dict and training history for plotting.
    """
    model = build_model_with_params(
        model_type, input_shape,
        best_params['units_1'], best_params['units_2'],
        best_params['dropout'], best_params['lr']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, validation_split=0.2,
        epochs=epochs, batch_size=best_params['batch_size'],
        callbacks=[early_stop], verbose=1
    )

    # Inverse transform predictions and targets back to real Wh scale
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_true = target_scaler.inverse_transform(y_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled)

    # Metric computations
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    mask = y_true.flatten() != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Save model binary
    os.makedirs('models', exist_ok=True)
    model.save(f'models/{model_type.lower().replace("-", "_")}_optimized.keras')

    metrics_result = {
        'model': f'{model_type} (Optimized)', 
        'MAE': mae, 
        'RMSE': rmse, 
        'MAPE': mape, 
        'R2': r2,
        'best_params': str(best_params)
    }

    return metrics_result, history.history


def optimize_all_models(train_path, test_path, seq_length=12, n_trials=10):
    """
    Full Optimization Pipeline across all 4 deep learning architectures.
    """
    from dlmodel import load_Sequences  # Ensure correct script name/import path

    X_train, y_train, X_test, y_test, target_scaler = load_Sequences(train_path, test_path, seq_length)

    # Carve out a validation split from training set for tuning phase
    val_idx = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:val_idx], X_train[val_idx:]
    y_tr, y_val = y_train[:val_idx], y_train[val_idx:]

    input_shape = (X_train.shape[1], X_train.shape[2])
    model_types = ['LSTM', 'GRU', 'CNN-LSTM_v2', 'Bi-LSTM']

    optimized_results = []
    all_histories = {}

    for model_type in model_types:
        best_params = tune_model(model_type, X_tr, y_tr, X_val, y_val, input_shape, n_trials=n_trials)
        
        result, history_dict = train_final_and_evaluate(
            model_type, best_params, X_train, y_train, X_test, y_test,
            input_shape, target_scaler
        )
        
        optimized_results.append(result)
        all_histories[model_type] = history_dict
        
        print(f"\n{model_type} (Optimized) — R2: {result['R2']:.4f}, MAE: {result['MAE']:.4f}")

    # Export results CSV for reporting and metrics tables
    results_df = pd.DataFrame(optimized_results)
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/deep_learning_optimized_results.csv', index=False)

    # Save training histories as JSON for loss curve plotting in plots.ipynb
    with open('reports/optimized_histories.json', 'w') as f:
        json.dump(all_histories, f)

    print("\n" + "=" * 60)
    print("OPTIMIZED MODELS — FINAL COMPARISON")
    print("=" * 60)
    print(results_df[['model', 'MAE', 'RMSE', 'MAPE', 'R2']].to_string(index=False))

    return results_df


if __name__ == '__main__':
    train_path = r'C:\Users\ASUS\OneDrive\Desktop\AI Intern Assessment\data\scaledfinal\train_nn_scaled_final.csv'
    test_path = r'C:\Users\ASUS\OneDrive\Desktop\AI Intern Assessment\data\scaledfinal\test_nn_scaled_final.csv'

   
    optimize_all_models(train_path, test_path, seq_length=12, n_trials=10)