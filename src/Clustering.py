"""
Clustering module for book categorization.
Implements K-Means clustering with feature engineering and model persistence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.sparse import hstack, csr_matrix
import joblib
from typing import Tuple, List


class BookClusterer:
    """
    Main class for clustering books based on text and numerical features.
    """
    
    def __init__(self, n_clusters: int = 5, max_features: int = 10000):
        """
        Initialize the clusterer.
        
        Args:
            n_clusters: Number of clusters for K-Means
            max_features: Maximum features for TF-IDF vectorization
        """
        self.n_clusters = n_clusters
        self.max_features = max_features
        
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, 2)  # Include bigrams for better context
        )
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        self.X_text = None
        self.X_num = None
        self.X_combined = None
        
    def fit_transform_features(self, df: pd.DataFrame) -> csr_matrix:
        """
        Create and transform text and numerical features.
        
        Args:
            df: DataFrame with 'text', 'avg_rating', 'num_rating' columns
            
        Returns:
            Combined feature matrix
        """
        print("Transforming text features with TF-IDF...")
        self.X_text = self.tfidf.fit_transform(df["text"])
        
        print("Scaling numerical features...")
        self.X_num = self.scaler.fit_transform(
            df[['avg_rating', 'num_rating']].values
        )
        
        print("Combining features...")
        self.X_combined = hstack([self.X_text, self.X_num])
        
        return self.X_combined
    
    def fit(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the K-Means model on the data.
        
        Args:
            df: DataFrame with book data
            
        Returns:
            Cluster labels
        """
        X = self.fit_transform_features(df)
        
        print(f"Fitting K-Means with {self.n_clusters} clusters...")
        labels = self.kmeans.fit_predict(X)
        
        return labels
    
    def get_top_terms_per_cluster(self, n_terms: int = 10) -> dict:
        """
        Get the top terms for each cluster.
        
        Args:
            n_terms: Number of top terms to retrieve
            
        Returns:
            Dictionary mapping cluster ID to top terms
        """
        order_centroids = self.kmeans.cluster_centers_[:, :self.X_text.shape[1]].argsort()[:, ::-1]
        terms = self.tfidf.get_feature_names_out()
        
        cluster_terms = {}
        for i in range(self.n_clusters):
            top_terms = [terms[ind] for ind in order_centroids[i, :n_terms]]
            cluster_terms[i] = top_terms
            
        return cluster_terms
    
    def save_models(self, model_dir: str = "../models"):
        """
        Save all trained models to disk.
        
        Args:
            model_dir: Directory to save models
        """
        print(f"Saving models to {model_dir}...")
        joblib.dump(self.tfidf, f"{model_dir}/tfidf_vectorizer.pkl")
        joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
        joblib.dump(self.kmeans, f"{model_dir}/kmeans_model.pkl")
        print("Models saved successfully!")
    
    @classmethod
    def load_models(cls, model_dir: str = "../models"):
        """
        Load saved models from disk.
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            BookClusterer instance with loaded models
        """
        clusterer = cls()
        clusterer.tfidf = joblib.load(f"{model_dir}/tfidf_vectorizer.pkl")
        clusterer.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        clusterer.kmeans = joblib.load(f"{model_dir}/kmeans_model.pkl")
        clusterer.n_clusters = clusterer.kmeans.n_clusters
        
        return clusterer


def find_optimal_k(df: pd.DataFrame, k_range: range = range(2, 15)) -> Tuple[List[int], List[float]]:
    """
    Use the elbow method to find optimal number of clusters.
    
    Args:
        df: DataFrame with book data
        k_range: Range of k values to test
        
    Returns:
        Tuple of (k_values, inertia_values)
    """
    # Create features
    clusterer = BookClusterer()
    X = clusterer.fit_transform_features(df)
    
    inertia = []
    k_values = list(k_range)
    
    print("Finding optimal k using elbow method...")
    for k in k_values:
        print(f"Testing k={k}...")
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertia.append(km.inertia_)
    
    return k_values, inertia


def plot_elbow_curve(k_values: List[int], inertia: List[float], save_path: str = None):
    """
    Plot the elbow curve for K-Means clustering.
    
    Args:
        k_values: List of k values tested
        inertia: List of inertia values
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia, marker='o', linewidth=2, markersize=8)
    plt.xlabel("Number of Clusters (k)", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.title("Elbow Method for Optimal K", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def assign_cluster_labels(cluster_terms: dict) -> dict:
    """
    Manually assign meaningful labels to clusters based on top terms.
    
    Args:
        cluster_terms: Dictionary of cluster IDs to top terms
        
    Returns:
        Dictionary mapping cluster ID to category name
    """
    # This is a simplified version - you may need to adjust based on your data
    cluster_labels = {
        0: "Mystery & Thriller",
        1: "Horror & Suspense",
        2: "Classic Fiction & Fantasy",
        3: "Fantasy & Sci-Fi",
        4: "Young Adult & Adventure"
    }
    
    return cluster_labels


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("../data/goodreads_books_cleaned.csv")
    
    # Find optimal k
    k_values, inertia = find_optimal_k(df)
    plot_elbow_curve(k_values, inertia)
    
    # Train final model
    clusterer = BookClusterer(n_clusters=5)
    labels = clusterer.fit(df)
    
    # Add cluster labels to dataframe
    df['cluster'] = labels
    
    # Get and print top terms
    cluster_terms = clusterer.get_top_terms_per_cluster()
    for i, terms in cluster_terms.items():
        print(f"\nCluster {i} top terms: {', '.join(terms)}")
    
    # Assign category labels
    category_labels = assign_cluster_labels(cluster_terms)
    df['category'] = df['cluster'].map(category_labels)
    
    # Save models and data
    clusterer.save_models()
    df.to_csv("../data/goodreads_books_clustered.csv", index=False)
    
    print("\nClustering complete!")
