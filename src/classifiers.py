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
    """k-NN with GridSearchCV over k.

    Two steps before training:
      1. Class balancing via oversampling - k-NN doesn't support class_weight,
         so on an 80/20 dataset it would always predict the majority class.
         Balancing makes the comparison with other models fair.
      2. Stratified subsample of 20K rows - k-NN is slow on tens of thousands
         of rows (distance computation for every point).
    """
    print('\n' + '-' * 60)
    print('  1. k-NN (K-Nearest Neighbors) - distance-based')
    print('-' * 60)

    # 1. Balance the classes (same approach as NN, since k-NN lacks class_weight)
    X_bal, y_bal = _balance_oversampling(X_train, y_train)
    print(f'  Data balanced via oversampling: {len(X_bal):,} rows '
          f'(50/50 attended/no-show)')

    # 2. Stratified subsample for speed
    from sklearn.model_selection import train_test_split as _tts
    if len(X_bal) > 20000:
        X_knn, _, y_knn, _ = _tts(X_bal, y_bal, train_size=20000,
                                  stratify=y_bal, random_state=42)
        print(f'  k-NN trained on a subsample of {len(X_knn):,} rows '
              f'(out of {len(X_bal):,}) for efficiency.')
    else:
        X_knn, y_knn = X_bal, y_bal

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

def train_logistic(X_train, y_train):
    """Logistic Regression with GridSearchCV over C (regularization strength).
    Uses class_weight='balanced' to handle the imbalanced dataset."""
    print('\n' + '-' * 60)
    print('  3. Logistic Regression - linear approach')
    print('-' * 60)

    param_grid = {
        'C'      : [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver' : ['lbfgs']
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        param_grid, cv=3, scoring='f1', n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f'  Tested values   : C={param_grid["C"]}')
    print(f'  Best parameters : {grid.best_params_}')
    print(f'  Best CV F1      : {grid.best_score_:.4f}')
    return grid.best_estimator_

def _balance_oversampling(X_train, y_train, random_state=42):
    """MLPClassifier doesn't support class_weight, so we balance the classes
    manually by oversampling the minority class up to the majority size."""
    rng = np.random.RandomState(random_state)
    idx_maj = y_train[y_train == 0].index
    idx_min = y_train[y_train == 1].index
    idx_min_over = rng.choice(idx_min, size=len(idx_maj), replace=True)
    idx_bal = np.concatenate([idx_maj.values, idx_min_over])
    rng.shuffle(idx_bal)
    return X_train.loc[idx_bal], y_train.loc[idx_bal]


def train_neural_network(X_train, y_train):
    """Train two different NN architectures and return both.

    Architecture A (shallow): one hidden layer with 16 units
    Architecture B (deep)   : two hidden layers with 64 and 32 units

    Both use ReLU activation and the Adam optimizer.
    Trained on oversampled (balanced) data since MLP doesn't support class_weight.
    """
    print('\n' + '-' * 60)
    print('  4. Neural Network (MLP) - neural network approach')
    print('-' * 60)

    # Balance the classes for NN training
    X_bal, y_bal = _balance_oversampling(X_train, y_train)
    print(f'  Data balanced via oversampling: {len(X_bal):,} rows '
          f'(50/50 attended/no-show)')

    # Architecture A: shallow
    print('\n  Architecture A: (16) - one hidden layer')
    nn_a = MLPClassifier(hidden_layer_sizes=(16,), activation='relu',
                         solver='adam', max_iter=300, early_stopping=True,
                         random_state=42)
    nn_a.fit(X_bal, y_bal)
    print(f'     Layers     : Input -> 16 (ReLU) -> Output')
    print(f'     Iterations : {nn_a.n_iter_}')

    # Architecture B: deep
    print('\n  Architecture B: (64, 32) - two hidden layers')
    nn_b = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                         solver='adam', max_iter=300, early_stopping=True,
                         random_state=42)
    nn_b.fit(X_bal, y_bal)
    print(f'     Layers     : Input -> 64 (ReLU) -> 32 (ReLU) -> Output')
    print(f'     Iterations : {nn_b.n_iter_}')

    # Hyperparameter tuning on architecture B (alpha = regularization)
    print('\n  Hyperparameter tuning on architecture B (alpha):')
    param_grid = {'alpha': [0.0001, 0.001, 0.01]}
    grid = GridSearchCV(
        MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                      solver='adam', max_iter=300, early_stopping=True,
                      random_state=42),
        param_grid, cv=3, scoring='f1', n_jobs=-1
    )
    grid.fit(X_bal, y_bal)
    print(f'     Tested values   : alpha={param_grid["alpha"]}')
    print(f'     Best parameters : {grid.best_params_}')
    print(f'     Best CV F1      : {grid.best_score_:.4f}')

    return nn_a, grid.best_estimator_
