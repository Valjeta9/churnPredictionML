import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score


def find_optimal_k(X, k_range=range(2, 8), sample_size=10000):
    """Compute inertia (Elbow) and silhouette for several values of k.
    Silhouette is computed on a sample (sample_size) for speed.
    Returns (DataFrame [k, Inertia, Silhouette], k_best)."""
    print('\n' + '=' * 60)
    print('  SEARCHING FOR THE OPTIMAL NUMBER OF CLUSTERS (k)')
    print('=' * 60)

    # Sample for silhouette (expensive on 110K rows)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), min(sample_size, len(X)), replace=False)
    X_sample = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]

    results = []
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X_sample, km.predict(X_sample))
        results.append({'k': k, 'Inertia': round(km.inertia_, 1),
                        'Silhouette': round(sil, 4)})
        print(f'   k={k} -> Inertia={km.inertia_:>14.1f} | Silhouette={sil:.4f}')

    df = pd.DataFrame(results)
    k_best = int(df.loc[df['Silhouette'].idxmax(), 'k'])
    print(f'\n  k with the highest silhouette: k={k_best}')
    return df, k_best


def save_elbow_silhouette(df_k, output_path):
    """Plot Elbow and Silhouette curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle('Figure. Choosing the Number of Clusters (k)',
                 fontsize=13, fontweight='bold')
    axes[0].plot(df_k['k'], df_k['Inertia'], 'o-', color='#3498db', linewidth=2)
    axes[0].set_title('Elbow Method (Inertia)')
    axes[0].set_xlabel('Number of clusters (k)')
    axes[0].set_ylabel('Inertia (WCSS)')
    axes[0].grid(alpha=0.3)
    axes[1].plot(df_k['k'], df_k['Silhouette'], 'o-', color='#e74c3c', linewidth=2)
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('Number of clusters (k)')
    axes[1].set_ylabel('Silhouette (higher is better)')
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Elbow/Silhouette figure saved: {output_path}')

def apply_kmeans(X, y_true, k=2, output_path='results/clustering.png',
                 sample_size=8000):
    """Apply K-Means with the given k, visualize with PCA, and compare
    with the true labels. Returns a dict with labels, ari, crosstab."""
    print('\n' + '=' * 60)
    print(f'  K-MEANS with k={k}')
    print('=' * 60)

    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X)

    ari = adjusted_rand_score(y_true, cluster_labels)
    print(f'\n  Adjusted Rand Index (ARI) vs true labels: {ari:.4f}')
    print('  (ARI=1 perfect match, ARI~0 random match)')

    crosstab = pd.crosstab(cluster_labels, y_true,
                           rownames=['Cluster'], colnames=['True NoShow'])
    print('\n  Cross-tab (Cluster vs True NoShow):')
    print('  ' + crosstab.to_string().replace('\n', '\n  '))

    # PCA visualization (on a sample for plotting speed)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), min(sample_size, len(X)), replace=False)
    X_s = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
    cl_s = cluster_labels[idx]
    y_s  = y_true.iloc[idx].values if hasattr(y_true, 'iloc') else y_true[idx]

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_s)
    var_exp = pca.explained_variance_ratio_.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f'Figure. K-Means clustering (k={k}) - PCA visualization',
                 fontsize=13, fontweight='bold')
    s1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cl_s, cmap='viridis',
                         alpha=0.5, s=12)
    axes[0].set_title(f'K-Means clusters (explained variance: {var_exp:.1f}%)')
    axes[0].set_xlabel('Principal Component 1 (PC1)')
    axes[0].set_ylabel('Principal Component 2 (PC2)')
    plt.colorbar(s1, ax=axes[0], label='Cluster')
    s2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_s, cmap='coolwarm',
                         alpha=0.5, s=12)
    axes[1].set_title('True labels (NoShow: red=Yes, blue=No)')
    axes[1].set_xlabel('Principal Component 1 (PC1)')
    axes[1].set_ylabel('Principal Component 2 (PC2)')
    plt.colorbar(s2, ax=axes[1], label='NoShow')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n  Clustering figure saved: {output_path}')

    return {'labels': cluster_labels, 'ari': ari, 'crosstab': crosstab}