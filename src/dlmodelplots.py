import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model



train_path = r"C:\Users\ASUS\OneDrive\Desktop\AI Intern Assessment\data\scaledfinal\train_nn_scaled_final.csv"
test_path = r"C:\Users\ASUS\OneDrive\Desktop\AI Intern Assessment\data\scaledfinal\test_nn_scaled_final.csv"
model_dir = r"C:\Users\ASUS\OneDrive\Desktop\AI Intern Assessment\models"
reports_dir =r"C:\Users\ASUS\OneDrive\Desktop\AI Intern Assessment\reports"
os.makedirs(reports_dir, exist_ok=True)

# --- load data ---
train_df = pd.read_csv(train_path, index_col='date', parse_dates=True)
test_df = pd.read_csv(test_path, index_col='date', parse_dates=True)

feature_cols = [c for c in train_df.columns if c not in ['Appliances', 'Appliances_log']]
seq_length = 12

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# --- reload the SAME scaler used during training ---
target_scaler = joblib.load(f"{model_dir}/dl_target_scaler.pkl")
test_target_scaled = target_scaler.transform(test_df[['Appliances']])
X_test_seq, y_test_seq = create_sequences(test_df[feature_cols].values, test_target_scaled, seq_length)

y_true_real = target_scaler.inverse_transform(y_test_seq).flatten()

print(f"X_test_seq shape: {X_test_seq.shape}")

# --- load each optimized model and predict ---
model_files = {
    'LSTM': 'lstm_optimized.keras',
    'GRU': 'gru_optimized.keras',
    'CNN-LSTM_v2': 'cnn_lstm_v2_optimized.keras',
    'Bi-LSTM': 'bi_lstm_optimized.keras',
}

dl_predictions = {}
for model_name, filename in model_files.items():
    model = load_model(f"{model_dir}/{filename}")
    y_pred_scaled = model.predict(X_test_seq, verbose=0)
    dl_predictions[model_name] = target_scaler.inverse_transform(y_pred_scaled).flatten()
    print(f"{model_name}: predictions done")

# --- Plot 1: Predicted vs Actual, all 4 models ---
colors_dl = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, model_name in enumerate(dl_predictions.keys()):
    y_pred = dl_predictions[model_name]
    axes[i].scatter(y_true_real, y_pred, alpha=0.3, s=8, color=colors_dl[i])
    min_v, max_v = min(y_true_real.min(), y_pred.min()), max(y_true_real.max(), y_pred.max())
    axes[i].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
    axes[i].set_title(f'{model_name} (Optimized): Predicted vs Actual', fontweight='bold', fontsize=12)
    axes[i].set_xlabel('Actual Energy Consumption (Wh)')
    axes[i].set_ylabel('Predicted Energy Consumption (Wh)')
    axes[i].legend()
    axes[i].grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Deep Learning Models (Optimized): Predicted vs Actual', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{reports_dir}/dl_predicted_vs_actual.png", dpi=300, bbox_inches='tight')
print(f"Saved: {reports_dir}/dl_predicted_vs_actual.png")

# --- Plot 2: Residuals over time, all 4 models ---
test_dates = test_df.index[seq_length:]
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, model_name in enumerate(dl_predictions.keys()):
    y_pred = dl_predictions[model_name]
    residuals = y_true_real - y_pred
    axes[i].plot(test_dates, residuals, alpha=0.6, linewidth=0.8, color=colors_dl[i])
    axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    axes[i].set_title(f'{model_name} (Optimized): Residuals Over Time', fontweight='bold', fontsize=12)
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Residual (Actual - Predicted, Wh)')
    axes[i].grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Deep Learning Models (Optimized): Residuals Over Time', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{reports_dir}/dl_residuals.png", dpi=300, bbox_inches='tight')
print(f"Saved: {reports_dir}/dl_residuals.png")

# --- residual summary ---
print("=" * 60)
print("RESIDUAL SUMMARY — DEEP LEARNING MODELS (OPTIMIZED)")
print("=" * 60)
for model_name in dl_predictions.keys():
    residuals = y_true_real - dl_predictions[model_name]
    print(f"{model_name:15s} | mean: {residuals.mean():7.3f} | std: {residuals.std():7.3f}")