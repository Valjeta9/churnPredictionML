"""
feature_selection.py - Feature selection experiment
Project: Patient No-Show Prediction

Tests whether using FEWER features keeps or improves performance.
Method: SelectKBest with ANOVA F-test.
Reference model: Decision Tree (fast on 110K rows).
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


def rank_features(X_train, y_train, names):
    """Rank features by ANOVA F-score. Returns a DataFrame [Feature, F_Score]."""
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X_train, y_train)
    return pd.DataFrame({
        'Feature': names, 'F_Score': selector.scores_.round(2)
    }).sort_values('F_Score', ascending=False).reset_index(drop=True)


def run_feature_selection(X_train, X_test, y_train, y_test, names):
    """Compare performance using all features vs. top-K features.
    Returns (comparison DataFrame, ranking DataFrame)."""
    print('\n' + '=' * 60)
    print('  FEATURE SELECTION EXPERIMENT')
    print('=' * 60)

    ranking = rank_features(X_train, y_train, names)
    print('\n  Feature ranking by importance (ANOVA F-score):')
    print('  ' + ranking.to_string(index=False).replace('\n', '\n  '))

    n_total = len(names)
    k_values = sorted(set([3, 5, 8, n_total]))
    results = []
    for k in k_values:
        top_k = ranking.head(k)['Feature'].tolist()
        model = DecisionTreeClassifier(max_depth=7, class_weight='balanced',
                                       random_state=42)
        model.fit(X_train[top_k], y_train)
        f1 = f1_score(y_test, model.predict(X_test[top_k]))
        results.append({'Num_Features': k, 'F1-Score': round(f1, 4)})
        print(f'   k={k:>2} features -> F1 = {f1:.4f}')

    df_res = pd.DataFrame(results)
    best = df_res.loc[df_res['F1-Score'].idxmax()]
    print(f'\n  Best configuration: {int(best["Num_Features"])} features '
          f'(F1 = {best["F1-Score"]})')
    print('  Note: if F1 with fewer features is close to F1 with all features,')
    print('        reducing features simplifies the model without losing accuracy.')
    return df_res, ranking