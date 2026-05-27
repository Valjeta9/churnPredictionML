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

def train_knn(X_train, y_train):
    """k-NN with GridSearchCV over k, weights and metric.
    Trained on a stratified 20K subsample because k-NN is slow on 88K rows."""
    print('\n' + '-' * 60)
    print('  1. k-NN (K-Nearest Neighbors) - distance-based')
    print('-' * 60)

    # Stratified subsample for speed
    from sklearn.model_selection import train_test_split as _tts
    if len(X_train) > 20000:
        X_knn, _, y_knn, _ = _tts(X_train, y_train, train_size=20000,
                                  stratify=y_train, random_state=42)
        print(f'  k-NN trained on a subsample of {len(X_knn):,} rows '
              f'(out of {len(X_train):,}) for efficiency.')
    else:
        X_knn, y_knn = X_train, y_train

    param_grid = {
        'n_neighbors': [21, 31, 41]
    }
    grid = GridSearchCV(KNeighborsClassifier(metric='euclidean', weights='distance'),
                        param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_knn, y_knn)

    print(f'  Tested values   : k={param_grid["n_neighbors"]}, '
          f'weights=distance, metric=euclidean')
    print(f'  Best parameters : {grid.best_params_}')
    print(f'  Best CV F1      : {grid.best_score_:.4f}')
    return grid.best_estimator_

def train_decision_tree(X_train, y_train):
    """Decision Tree with GridSearchCV. Uses class_weight='balanced'
    to handle the imbalanced dataset."""
    print('\n' + '-' * 60)
    print('  2. Decision Tree - tree-based')
    print('-' * 60)

    param_grid = {
        'criterion'        : ['gini', 'entropy'],
        'max_depth'        : [3, 5, 7, 10, 15],
        'min_samples_leaf' : [1, 10, 50, 100]
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        param_grid, cv=3, scoring='f1', n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f'  Tested values   : criterion={param_grid["criterion"]}, '
          f'max_depth={param_grid["max_depth"]}, '
          f'min_samples_leaf={param_grid["min_samples_leaf"]}')
    print(f'  Best parameters : {grid.best_params_}')
    print(f'  Best CV F1      : {grid.best_score_:.4f}')
    return grid.best_estimator_