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
    # Binary target
    df['NoShow'] = (df['No-show'] == 'Yes').astype(int)
    # Remove negative age
    before = len(df)
    df = df[df['Age'] >= 0].copy()
    print(f'\nCleaning: {before - len(df)} row(s) with Age<0 removed.')
    print(f'Missing values: {df.isnull().sum().sum()}')
    print(f'Remaining rows: {len(df):,}')
    return df