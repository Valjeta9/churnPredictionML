import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Allows running from anywhere
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(BASE, 'src'))

from data_prep import load_data, clean_and_process, prepare_for_ml
from classifiers import (
    train_knn, train_decision_tree, train_logistic,
    train_neural_network, evaluate_model
)
from feature_selection import run_feature_selection
from clustering import find_optimal_k, save_elbow_silhouette, apply_kmeans
from sklearn.metrics import f1_score

DATA_PATH   = os.path.join(BASE, 'data', 'full_dataset.csv')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_confusion_matrices(results, output_path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.0))
    if n == 1:
        axes = [axes]
    fig.suptitle('Confusion Matrices - Classifier Comparison',
                 fontsize=14, fontweight='bold')
    for ax, r in zip(axes, results):
        sns.heatmap(r['ConfMatrix'], annot=True, fmt='d', cmap='Blues',
                    cbar=False, ax=ax,
                    xticklabels=['Attended', 'No-Show'],
                    yticklabels=['Attended', 'No-Show'])
        ax.set_title(f"{r['Model']}\nF1={r['F1-Score']}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Confusion matrices saved: {output_path}')


def save_comparison_table(df_compare, output_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(df_compare))
    w = 0.2
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    for i, (m, c) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * w, df_compare[m], w, label=m, color=c, edgecolor='white')
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(df_compare['Model'], fontsize=9)
    ax.set_ylabel('Value')
    ax.set_title('Classifier Performance Comparison',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', ncol=4, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Comparison table saved: {output_path}')


def main():
    print('\n' + '#' * 60)
    print('  NO-SHOW PREDICTION - ML PROJECT')
    print('#' * 60)

    # 1-2: Load, clean, prepare
    df_raw = load_data(DATA_PATH)
    df = clean_and_process(df_raw)
    X_train, X_test, y_train, y_test, scaler, features = prepare_for_ml(df)

    # 3: Train 4 classifiers
    print('\n' + '#' * 60)
    print('  TRAINING CLASSIFIERS')
    print('#' * 60)
    knn  = train_knn(X_train, y_train)
    tree = train_decision_tree(X_train, y_train)
    logr = train_logistic(X_train, y_train)
    nn_a, nn_b = train_neural_network(X_train, y_train)

    # 4: Compare NN architectures
    print('\n  NN ARCHITECTURE COMPARISON:')
    f1_a = f1_score(y_test, nn_a.predict(X_test))
    f1_b = f1_score(y_test, nn_b.predict(X_test))
    print(f'     Architecture A (16)     -> F1 = {f1_a:.4f}')
    print(f'     Architecture B (64,32)  -> F1 = {f1_b:.4f}')
    nn_best = nn_b if f1_b >= f1_a else nn_a
    arch_best = '(64,32)' if f1_b >= f1_a else '(16)'
    print(f'     Best architecture: {arch_best}')
    print('     Note: a deeper architecture captures more complex relationships,')
    print('           but risks overfitting if the data is not large enough.')

    # 5: Feature selection
    df_fs, ranking = run_feature_selection(X_train, X_test, y_train, y_test, features)
    df_fs.to_csv(os.path.join(RESULTS_DIR, 'feature_selection.csv'), index=False)
    ranking.to_csv(os.path.join(RESULTS_DIR, 'feature_ranking.csv'), index=False)

    # 6: Evaluate + comparison
    print('\n' + '#' * 60)
    print('  MODEL EVALUATION AND COMPARISON')
    print('#' * 60)
    results = [
        evaluate_model('k-NN', knn, X_test, y_test),
        evaluate_model('Decision Tree', tree, X_test, y_test),
        evaluate_model('Logistic Reg.', logr, X_test, y_test),
        evaluate_model(f'Neural Net {arch_best}', nn_best, X_test, y_test),
    ]
    df_compare = pd.DataFrame([
        {k: r[k] for k in ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']}
        for r in results
    ])
    print('\n  CLASSIFIER COMPARISON TABLE:')
    print('  ' + df_compare.to_string(index=False).replace('\n', '\n  '))
    winner = df_compare.loc[df_compare['F1-Score'].idxmax(), 'Model']
    print(f'\n  Best classifier (by F1): {winner}')

    df_compare.to_csv(os.path.join(RESULTS_DIR, 'classifier_comparison.csv'), index=False)
    save_comparison_table(df_compare, os.path.join(RESULTS_DIR, 'comparison.png'))
    save_confusion_matrices(results, os.path.join(RESULTS_DIR, 'confusion_matrices.png'))

    # 7: Clustering
    print('\n' + '#' * 60)
    print('  CLUSTERING (unsupervised)')
    print('#' * 60)
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    df_k, k_best = find_optimal_k(X_all)
    df_k.to_csv(os.path.join(RESULTS_DIR, 'clustering_k_analysis.csv'), index=False)
    save_elbow_silhouette(df_k, os.path.join(RESULTS_DIR, 'clustering_elbow.png'))
    apply_kmeans(X_all, y_all, k=2, output_path=os.path.join(RESULTS_DIR, 'clustering.png'))

    print('\n' + '#' * 60)
    print('  PROJECT RAN SUCCESSFULLY')
    print('#' * 60)
    print(f'  All results saved in: {RESULTS_DIR}/')


if __name__ == '__main__':
    main()