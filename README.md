# Multivariate Time-Series Prediction of Household Appliance Energy Consumption

## Overview

This project predicts household appliance energy consumption (Appliances, measured in Wh)
using the UCI Appliances Energy Prediction dataset — 19,735 records collected at 10-minute
intervals over roughly 4.5 months (11 January to 27 May 2016), combining indoor sensor
readings across nine zones, outdoor weather data, and the energy consumption target.

The goal was not only to build an accurate predictive model but to handle the challenges
specific to time-series data: preventing data leakage, engineering features that respect the
sequential nature of the data, and comparing model complexity fairly. The pipeline evaluates
five baseline models (naive persistence, Decision Tree, Random Forest, XGBoost, Linear
Regression) alongside four deep learning architectures (LSTM, GRU, CNN-LSTM, Bidirectional
LSTM), with hyperparameter tuning applied to all four deep learning models via Optuna.

The strongest overall performer was Linear Regression on the raw target (R2 of 0.549). None
of the deep learning models closed that gap, since roughly 37 percent of the predictive
signal comes from a single feature, the previous 10-minute reading of the target itself — a
short-term persistence signal a linear model captures directly.

## Setup

1. Clone the repository
2.  Create and activate a virtual environment
3.  Install dependencies


## How to Run

The scripts must be run in this order, since each stage depends on the output of the one
before it. Run all commands from the project root.

1. Feature engineering — adds cyclical time features, holiday flag, indoor/outdoor
   interaction terms, and lag and rolling features
2. Data preprocessing — splits the data chronologically and produces unscaled and
   MinMax-scaled versions for different model families
3. Feature selection — selects the final feature set using Random Forest importance and
   Recursive Feature Elimination
4. Baseline models — trains and evaluates naive persistence, Decision Tree, Random Forest,
   XGBoost, and Linear Regression
5. Deep learning models — trains LSTM, GRU, CNN-LSTM, and Bidirectional LSTM
6. Hyperparameter optimization — tunes all four deep learning architectures with Optuna and
   re-evaluates on the test set
7. Report figures — generate the plots used in the final report
8. Exploratory data analysis — open and run the notebook directly:

## Notes
Deep learning results may vary slightly (from the metrics in the report) between runs due to GPU/cuDNN
non-determinism, even with random seeds set. The relative pattern reported, 
recurrent architectures clustering below Linear Regression, is consistent across runs.
