# data_processing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def load_and_preprocess_data():
    df = pd.read_csv('creditcard.csv')

    # --- Data Visualization ---
    print("Initial Class Distribution:")
    sns.countplot(x='Class', data=df)
    plt.title('Initial Class Distribution (Imbalanced)')
    plt.show()

    # --- Data Preprocessing ---
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Handle imbalance with SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print("Data preprocessed and ready for training.")
    return X_resampled, y_resampled, X_test, y_test

if __name__ == '__main__':
    load_and_preprocess_data()