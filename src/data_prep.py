"""
data_prep.py - Data loading, cleaning and preprocessing
Project: Patient No-Show Prediction
Dataset: Medical Appointment No-Shows (Kaggle)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    """Load the No-Show dataset from a CSV file and print some basic info."""
    df = pd.read_csv(path)
    print('=' * 60)
    print('  BASIC DATASET INFORMATION')
    print('=' * 60)
    print(f'  Rows (appointments) : {df.shape[0]:,}')
    print(f'  Columns (features)  : {df.shape[1]}')
    print(f'  Memory usage        : {df.memory_usage(deep=True).sum()/1024**2:.2f} MB')
    print('=' * 60)
    return df


def clean_and_process(df):
    """Clean the data and add a few new features (waiting days, day of week,
    appointment hour, binary target). Also drops rows with negative age."""
    df = df.copy()
    # Convert dates
    df['ScheduledDay']   = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    # Feature engineering
    df['WaitingDays']    = (df['AppointmentDay'] - df['ScheduledDay']).dt.days.clip(lower=0)
    df['AppDayOfWeek']   = df['AppointmentDay'].dt.dayofweek
    df['AppHour']        = df['ScheduledDay'].dt.hour
    df['ScheduledMonth'] = df['ScheduledDay'].dt.month
    # Binary target
    df['NoShow'] = (df['No-show'] == 'Yes').astype(int)
    # Remove negative age
    before = len(df)
    df = df[df['Age'] >= 0].copy()
    print(f'\nCleaning: {before - len(df)} row(s) with Age<0 removed.')
    print(f'Missing values: {df.isnull().sum().sum()}')
    print(f'Remaining rows: {len(df):,}')
    return df

def prepare_for_ml(df, target='NoShow', test_size=0.2, random_state=42):
    """Encode features, scale them, and split into train/test sets.
    Returns: X_train, X_test, y_train, y_test, scaler, feature_names."""
    df = df.copy()

    # Encode Gender as binary
    df['Gender'] = (df['Gender'] == 'F').astype(int)   # F=1, M=0

    # Frequency encoding for Neighbourhood (81 categories)
    freq = df['Neighbourhood'].value_counts(normalize=True)
    df['Neighbourhood_freq'] = df['Neighbourhood'].map(freq)

    # Features used by the models
    features = [
        'Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes',
        'Alcoholism', 'Handcap', 'SMS_received',
        'WaitingDays', 'AppDayOfWeek', 'AppHour', 'ScheduledMonth',
        'Neighbourhood_freq'
    ]

    X = df[features].copy()
    y = df[target].copy()

    print(f'\nSelected features ({len(features)}): {features}')

    # Train/test split with stratify (keeps the 80/20 class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scaling - fit ONLY on train to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=features, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=features, index=X_test.index
    )

    print(f'\nTrain/test split:')
    print(f'  Train: {X_train_scaled.shape[0]:,} rows')
    print(f'  Test : {X_test_scaled.shape[0]:,} rows')
    print(f'  No-Show ratio (train): {y_train.mean()*100:.1f}%')
    print(f'  No-Show ratio (test) : {y_test.mean()*100:.1f}%')
    print('  Note: imbalanced dataset -> we will use F1-Score, not just Accuracy')

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features