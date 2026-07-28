import pandas as pd
import numpy as np
import holidays


#adding sin/cos encoing instead of raw intergers. 
def add_cyclical_time_features(df):
    df=df.copy()
    #extracting raw dattime components
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    #continous cyclical sin/cosin trans
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int) #1 if weekend 0 is weekday
    return df

def add_holiday_flag(df, country='SriLanka'):
    df = df.copy()
    years = df.index.year.unique().tolist()
    country_holidays = getattr(holidays, country)(years=years)
    df['is_holiday'] = df.index.normalize().isin(country_holidays).astype(int)
    return df

def add_lagged_features(df, target ='Appliances', lags =(1, 2, 3, 6, 18, 144)):
    df = df.copy()
    for lag in lags:
        df[f'{target}_lag_{lag}'] = df[target].shift(lag)
    return df

def add_rolling_features(df, target='Appliances', windows=(6, 18)):
    
    df = df.copy()
    for w in windows:
        label = '1hr' if w == 6 else '3hr' if w == 18 else f'{w}steps'
        df[f'{target}_roll_mean_{label}'] = df[target].shift(1).rolling(window=w).mean()
        df[f'{target}_roll_std_{label}'] = df[target].shift(1).rolling(window=w).std()
    return df

def add_indoor_outdoor_features(df):
    df = df.copy()
    indoor_temp_cols = [c for c in df.columns if c.startswith('T') and c[1:].isdigit()]
    indoor_rh_cols = [c for c in df.columns if c.startswith('RH_') and c[3:].isdigit()]

    df['T_indoor_avg'] = df[indoor_temp_cols].mean(axis=1)
    df['RH_indoor_avg'] = df[indoor_rh_cols].mean(axis=1)
    df['temp_diff'] = df['T_indoor_avg'] - df['T_out']
    df['T_indoor_RH_interaction'] = df['T_indoor_avg'] * df['RH_indoor_avg']
    df['T_out_RH_out_interaction'] = df['T_out'] * df['RH_out']
    return df

def add_lights_flag(df):
    df = df.copy()
    df['lights_on'] = (df['lights'] > 0).astype(int)
    return df

def engineer_all_features(df, target='Appliances'):
 
    df = add_cyclical_time_features(df)
    df = add_holiday_flag(df)
    df = add_indoor_outdoor_features(df)
    df = add_lights_flag(df)
    df = add_lagged_features(df, target=target)   # raw-Appliances lags, for tree models
    df = add_rolling_features(df, target=target)  # raw-Appliances rolling stats

    df.dropna(inplace=True) 
    return df

if __name__ == '__main__':
    df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\AI Intern Assessment\data\raw.csv', parse_dates=['date'], index_col='date')
    df_engineered = engineer_all_features(df, target='Appliances')
    df_engineered.to_csv(r'C:\Users\ASUS\OneDrive\Desktop\AI Intern Assessment\data\engineeredfeatures.csv')
    print(f"Final shape: {df_engineered.shape}")