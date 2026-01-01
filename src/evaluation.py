"""
Model evaluation module for clustering analysis.
Provides metrics and visualizations for cluster quality assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_clustering_metrics(X: csr_matrix, labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate various clustering evaluation metrics.
    
    Args:
        X: Feature matrix (sparse or dense)
        labels: Cluster labels
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    print("Calculating clustering metrics...")
    
    # Silhouette Score (higher is better, range: -1 to 1)
    silhouette = silhouette_score(X, labels, sample_size=10000)
    metrics['silhouette_score'] = silhouette
    
    # Davies-Bouldin Index (lower is better, 0 to infinity)
    davies_bouldin = davies_bouldin_score(X.toarray() if hasattr(X, 'toarray') else X, labels)
    metrics['davies_bouldin_index'] = davies_bouldin
    
    # Calinski-Harabasz Score (higher is better)
    calinski = calinski_harabasz_score(X.toarray() if hasattr(X, 'toarray') else X, labels)
    metrics['calinski_harabasz_score'] = calinski
    
    return metrics


def print_metrics(metrics: Dict[str, float]):
    """
    Print clustering metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
    """
    print("\n" + "="*50)
    print("CLUSTERING EVALUATION METRICS")
    print("="*50)
    
    print(f"\n📊 Silhouette Score: {metrics['silhouette_score']:.4f}")
    print("   → Measures cluster cohesion and separation")
    print("   → Range: -1 (worst) to 1 (best)")
    print("   → > 0.5: Good, > 0.7: Excellent")
    
    print(f"\n📉 Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
    print("   → Measures cluster similarity (lower is better)")
    print("   → Range: 0 to infinity")
    print("   → < 1: Good separation")
    
    print(f"\n📈 Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f}")
    print("   → Ratio of between-cluster to within-cluster dispersion")
    print("   → Higher values indicate better-defined clusters")
    
    print("\n" + "="*50 + "\n")


def analyze_cluster_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the distribution of books across clusters.
    
    Args:
        df: DataFrame with 'cluster' column
        
    Returns:
        DataFrame with cluster statistics
    """
    cluster_stats = df.groupby('cluster').agg({
        'title': 'count',
        'avg_rating': ['mean', 'std', 'min', 'max'],
        'num_rating': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    cluster_stats = cluster_stats.rename(columns={'title_count': 'num_books'})
    
    return cluster_stats


def plot_cluster_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Plot the distribution of books across clusters.
    
    Args:
        df: DataFrame with 'cluster' and optionally 'category' columns
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Count books per cluster
    cluster_counts = df['cluster'].value_counts().sort_index()
    
    # Use category names if available
    if 'category' in df.columns:
        labels = [df[df['cluster'] == i]['category'].iloc[0] for i in cluster_counts.index]
    else:
        labels = [f"Cluster {i}" for i in cluster_counts.index]
    
    colors = plt.cm.Set3(range(len(cluster_counts)))
    bars = plt.bar(range(len(cluster_counts)), cluster_counts.values, color=colors)
    
    plt.xlabel('Cluster', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Books', fontsize=12, fontweight='bold')
    plt.title('Distribution of Books Across Clusters', fontsize=14, fontweight='bold')
    plt.xticks(range(len(cluster_counts)), labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_cluster_characteristics(df: pd.DataFrame, save_path: str = None):
    """
    Plot box plots showing rating and popularity characteristics per cluster.
    
    Args:
        df: DataFrame with cluster information
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average Rating by Cluster
    df_sorted = df.sort_values('cluster')
    if 'category' in df.columns:
        labels = [df[df['cluster'] == i]['category'].iloc[0] 
                 for i in sorted(df['cluster'].unique())]
        df_sorted['cluster_label'] = df_sorted['cluster'].map(
            {i: labels[i] for i in range(len(labels))}
        )
        x_col = 'cluster_label'
    else:
        x_col = 'cluster'
    
    sns.boxplot(data=df_sorted, x=x_col, y='avg_rating', ax=axes[0], palette='Set3')
    axes[0].set_xlabel('Cluster', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Average Rating', fontsize=12, fontweight='bold')
    axes[0].set_title('Average Rating Distribution by Cluster', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Number of Ratings by Cluster (log scale)
    sns.boxplot(data=df_sorted, x=x_col, y='num_rating', ax=axes[1], palette='Set3')
    axes[1].set_xlabel('Cluster', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Number of Ratings (log scale)', fontsize=12, fontweight='bold')
    axes[1].set_title('Popularity Distribution by Cluster', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_pca_clusters(X: csr_matrix, labels: np.ndarray, df: pd.DataFrame = None, save_path: str = None):
    """
    Reduce dimensions with PCA and plot clusters in 2D.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        df: Optional DataFrame with category names
        save_path: Optional path to save the plot
    """
    print("Reducing dimensions with PCA...")
    
    # Convert to dense if needed (sample for large datasets)
    if X.shape[0] > 10000:
        sample_indices = np.random.choice(X.shape[0], 10000, replace=False)
        X_sample = X[sample_indices].toarray() if hasattr(X, 'toarray') else X[sample_indices]
        labels_sample = labels[sample_indices]
    else:
        X_sample = X.toarray() if hasattr(X, 'toarray') else X
        labels_sample = labels
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sample)
    
    plt.figure(figsize=(12, 8))
    
    # Get unique clusters
    unique_labels = np.unique(labels_sample)
    colors = plt.cm.Set3(range(len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_sample == label
        
        if df is not None and 'category' in df.columns:
            cluster_name = df[df['cluster'] == label]['category'].iloc[0]
        else:
            cluster_name = f"Cluster {label}"
        
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[colors[i]], label=cluster_name, alpha=0.6, s=50)
    
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)', 
               fontsize=12, fontweight='bold')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)', 
               fontsize=12, fontweight='bold')
    plt.title('Book Clusters Visualized in 2D (PCA)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print(f"Total variance explained by 2 components: {pca.explained_variance_ratio_.sum():.2%}")


def get_top_books_per_cluster(df: pd.DataFrame, n_books: int = 5) -> Dict[int, pd.DataFrame]:
    """
    Get top rated books for each cluster.
    
    Args:
        df: DataFrame with cluster information
        n_books: Number of top books to retrieve per cluster
        
    Returns:
        Dictionary mapping cluster ID to top books DataFrame
    """
    top_books = {}
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        top = cluster_df.nlargest(n_books, 'avg_rating')[['title', 'author', 'avg_rating', 'num_rating']]
        top_books[cluster_id] = top
    
    return top_books


def print_top_books(top_books: Dict[int, pd.DataFrame], df: pd.DataFrame):
    """
    Print top books for each cluster in a formatted way.
    
    Args:
        top_books: Dictionary of cluster ID to top books DataFrame
        df: Original DataFrame with category information
    """
    print("\n" + "="*80)
    print("TOP BOOKS PER CLUSTER")
    print("="*80)
    
    for cluster_id, books in top_books.items():
        if 'category' in df.columns:
            category = df[df['cluster'] == cluster_id]['category'].iloc[0]
            print(f"\n📚 {category} (Cluster {cluster_id})")
        else:
            print(f"\n📚 Cluster {cluster_id}")
        
        print("-" * 80)
        for idx, row in books.iterrows():
            print(f"   • {row['title']}")
            print(f"     by {row['author']}")
            print(f"     ⭐ {row['avg_rating']:.2f} ({row['num_rating']:,} ratings)")
            print()


if __name__ == "__main__":
    # Example usage
    from clustering import BookClusterer
    
    # Load data and model
    df = pd.read_csv("../data/goodreads_books_clustered.csv")
    clusterer = BookClusterer.load_models()
    
    # Get features
    X = clusterer.fit_transform_features(df)
    labels = df['cluster'].values
    
    # Calculate and print metrics
    metrics = calculate_clustering_metrics(X, labels)
    print_metrics(metrics)
    
    # Analyze distribution
    cluster_stats = analyze_cluster_distribution(df)
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    # Visualizations
    plot_cluster_distribution(df)
    plot_cluster_characteristics(df)
    plot_pca_clusters(X, labels, df)
    
    # Top books
    top_books = get_top_books_per_cluster(df, n_books=5)
    print_top_books(top_books, df)
