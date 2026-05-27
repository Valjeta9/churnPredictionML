"""
classifiers.py - Four classifiers with hyperparameter tuning
Project: Patient No-Show Prediction

Four classifiers with different approaches:
  1. k-NN               -> distance-based
  2. Decision Tree      -> tree-based
  3. Logistic Regression -> linear
  4. Neural Network     -> neural network (2 architectures)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)


def evaluate_model(name, model, X_test, y_test):
    """Compute all evaluation metrics for a trained model.
    Returns a dict with Accuracy, Precision, Recall, F1, ConfMatrix, y_pred."""
    y_pred = model.predict(X_test)
    return {
        'Model'     : name,
        'Accuracy'  : round(accuracy_score(y_test, y_pred), 4),
        'Precision' : round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall'    : round(recall_score(y_test, y_pred, zero_division=0), 4),
        'F1-Score'  : round(f1_score(y_test, y_pred, zero_division=0), 4),
        'ConfMatrix': confusion_matrix(y_test, y_pred),
        'y_pred'    : y_pred
    }