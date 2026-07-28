# src/deep_learning_comparison_fixed.py

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, Conv1D, Dense, Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def load_Sequences(train_path, test_path, seq_length=12):
    print("=" * 60)
    print("LOADING DATA...")
    print("=" * 60)
    
    train = pd.read_csv(train_path, index_col='date', parse_dates=True)
    test = pd.read_csv(test_path, index_col='date', parse_dates=True)
    
    print(f"Training shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    feature_cols = [c for c in train.columns if c not in ['Appliances', 'Appliances_log']]
    target_col = 'Appliances'

    target_scaler = MinMaxScaler()
    train_target = target_scaler.fit_transform(train[[target_col]])
    test_target = target_scaler.transform(test[[target_col]])

    os.makedirs('models', exist_ok=True)
    joblib.dump(target_scaler, 'models/dl_target_scaler.pkl')

    def create_sequences(X, y, seq_len):
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i - seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    X_train, y_train = create_sequences(train[feature_cols].values, train_target, seq_length)
    X_test, y_test = create_sequences(test[feature_cols].values, test_target, seq_length)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("=" * 60)

    return X_train, y_train, X_test, y_test, target_scaler


def build_lstm(input_shape):
    """Standard LSTM with 2 layers (cuDNN enabled)"""
    model = Sequential([
        LSTM(64, return_sequences=True, dropout=0.2, input_shape=input_shape),
        LSTM(64, dropout=0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss=Huber(delta=1.0), metrics=['mae'])
    return model


def build_gru(input_shape):
    """Standard GRU with 2 layers (cuDNN enabled)"""
    model = Sequential([
        GRU(64, return_sequences=True, dropout=0.2, input_shape=input_shape),
        GRU(64, dropout=0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss=Huber(delta=1.0), metrics=['mae'])
    return model


def build_cnn_lstm_v2(input_shape):
    """Improved CNN-LSTM (cuDNN enabled)"""
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', padding='same', input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        LSTM(64, return_sequences=True, dropout=0.2),
        LSTM(64, dropout=0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss=Huber(delta=1.0), metrics=['mae'])
    return model


def build_bidirectional_lstm(input_shape):
    """Bidirectional LSTM (cuDNN enabled)"""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2), input_shape=input_shape),
        Bidirectional(LSTM(64, dropout=0.2)),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss=Huber(delta=1.0), metrics=['mae'])
    return model


# Model registry
MODEL_BUILDERS = {
    'LSTM': build_lstm,
    'GRU': build_gru,
    'CNN-LSTM_v2': build_cnn_lstm_v2,
    'Bi-LSTM': build_bidirectional_lstm,
}


def train_one_model(model_name, build_fn, X_train, y_train, X_test, y_test, target_scaler, epochs=100, batch_size=64):
    print("\n" + "=" * 60)
    print(f"{model_name} MODEL TRAINING")
    print("=" * 60)
    
    try:
        model = build_fn((X_train.shape[1], X_train.shape[2]))
        model.summary()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_true = target_scaler.inverse_transform(y_test)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mask = y_true.flatten() != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        print("\n" + "=" * 60)
        print(f"{model_name} RESULTS")
        print("=" * 60)
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²:   {r2:.4f}")
        print("=" * 60)
        
        # Save model
        fname = model_name.lower().replace('-', '_')
        model.save(f'models/{fname}_model.keras')
        print(f"Model saved to models/{fname}_model.keras")
        
        return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}, history.history
        
    except Exception as e:
        print(f" ERROR in {model_name}: {str(e)}")
        return None, None


def plot_comparison(results_df):
    """Create comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        bars = ax.bar(results_df['model'], results_df[metric], color=colors[:len(results_df)])
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Highlight best
        if metric == 'R2':
            best_idx = results_df[metric].argmax()
            bars[best_idx].set_color('#2ECC71')
            ax.text(0.5, 0.95, f'Best: {results_df.iloc[best_idx]["model"]}', 
                   transform=ax.transAxes, ha='center', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            best_idx = results_df[metric].argmin()
            bars[best_idx].set_color('#2ECC71')
            ax.text(0.5, 0.95, f'Best: {results_df.iloc[best_idx]["model"]}',
                   transform=ax.transAxes, ha='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/model_comparison.png', dpi=300)
    plt.show()
    print("Comparison plot saved to reports/model_comparison.png")


def run_all_models(train_path, test_path, seq_length=12, epochs=50, batch_size=64):
    """Run all deep learning architectures"""
    
    X_train, y_train, X_test, y_test, target_scaler = load_Sequences(
        train_path, test_path, seq_length
    )
    
    all_results = []
    histories = {}
    
    for model_name, build_fn in MODEL_BUILDERS.items():
        result, history_dict = train_one_model(
            model_name, build_fn, X_train, y_train, X_test, y_test,
            target_scaler, epochs=epochs, batch_size=batch_size
        )
        if result:
            all_results.append(result)
            histories[model_name] = history_dict
    
    # Create comparison
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
    
    # Highlight winners
    best_r2 = results_df.loc[results_df['R2'].idxmax()]
    print(f"\nBest Model: {best_r2['model']}")
    print(f"   R²: {best_r2['R2']:.4f}")
    print(f"   MAE: {best_r2['MAE']:.4f}")
    print(f"   MAPE: {best_r2['MAPE']:.2f}%")
    
    # Save results CSV & training histories JSON for notebook plotting
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/deep_learning_comparison.csv', index=False)
    
    with open('reports/initial_dl_histories.json', 'w') as f:
        json.dump(histories, f)
        
    print(f"\n Results saved to reports/deep_learning_comparison.csv & reports/initial_dl_histories.json")
    
    # Plot comparison
    plot_comparison(results_df)
    
    return results_df, histories


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    train_path = r'C:\Users\ASUS\OneDrive\Desktop\AI Intern Assessment\data\scaledfinal\train_nn_scaled_final.csv'
    test_path = r'C:\Users\ASUS\OneDrive\Desktop\AI Intern Assessment\data\scaledfinal\test_nn_scaled_final.csv'
    
    # Fixed seq_length=12 for accurate temporal context & CNN performance
    results_df, histories = run_all_models(
        train_path,
        test_path,
        seq_length=12,  
        epochs=50,      
        batch_size=64
    )