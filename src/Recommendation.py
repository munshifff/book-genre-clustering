"""
Recommendation engine for suggesting similar books based on clustering.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import joblib
from scipy.sparse import hstack


class BookRecommender:
    """
    Book recommendation system using cluster-based similarity.
    """
    
    def __init__(self, data_path: str = None, model_dir: str = None):
        """
        Initialize the recommender system.
        
        Args:
            data_path: Path to clustered books CSV
            model_dir: Path to directory containing saved models
        """
        if data_path is None:
            data_path = "../data/goodreads_books_clustered.csv"
        if model_dir is None:
            model_dir = "../models"
            
        # Load data
        self.df = pd.read_csv(data_path)
        
        # Load models
        self.tfidf = joblib.load(f"{model_dir}/tfidf_vectorizer.pkl")
        self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
        self.kmeans = joblib.load(f"{model_dir}/kmeans_model.pkl")
        
    def _find_book(self, query: str) -> Optional[pd.Series]:
        """
        Find a book by title (case-insensitive partial match).
        
        Args:
            query: Book title to search for
            
        Returns:
            Series with book data or None if not found
        """
        matches = self.df[self.df['title'].str.contains(query, case=False, na=False)]
        
        if len(matches) == 0:
            return None
        
        # Return the most popular match
        return matches.nlargest(1, 'num_rating').iloc[0]
    
    def recommend_by_title(self, title: str, n: int = 5, 
                          same_author: bool = False) -> pd.DataFrame:
        """
        Get book recommendations based on a book title.
        
        Args:
            title: Title of the book to base recommendations on
            n: Number of recommendations to return
            same_author: If True, exclude books by the same author
            
        Returns:
            DataFrame with recommended books
        """
        # Find the book
        book = self._find_book(title)
        
        if book is None:
            return pd.DataFrame()  # Empty DataFrame if not found
        
        # Get books from the same cluster
        cluster_id = book['cluster']
        cluster_books = self.df[self.df['cluster'] == cluster_id].copy()
        
        # Exclude the original book
        cluster_books = cluster_books[cluster_books['title'] != book['title']]
        
        # Optionally exclude same author
        if same_author:
            cluster_books = cluster_books[cluster_books['author'] != book['author']]
        
        # Sort by rating and popularity
        cluster_books['score'] = (
            cluster_books['avg_rating'] * 0.7 + 
            np.log1p(cluster_books['num_rating']) * 0.3
        )
        
        recommendations = cluster_books.nlargest(n, 'score')[
            ['title', 'author', 'avg_rating', 'num_rating', 'category']
        ]
        
        return recommendations.reset_index(drop=True)
    
    def recommend_by_author(self, author: str, n: int = 5) -> pd.DataFrame:
        """
        Get book recommendations based on an author's typical cluster.
        
        Args:
            author: Author name to search for
            n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended books
        """
        # Find books by this author
        author_books = self.df[self.df['author'].str.contains(author, case=False, na=False)]
        
        if len(author_books) == 0:
            return pd.DataFrame()
        
        # Get the most common cluster for this author
        cluster_id = author_books['cluster'].mode()[0]
        
        # Get books from this cluster (excluding this author)
        cluster_books = self.df[
            (self.df['cluster'] == cluster_id) & 
            (~self.df['author'].str.contains(author, case=False, na=False))
        ].copy()
        
        # Score and rank
        cluster_books['score'] = (
            cluster_books['avg_rating'] * 0.7 + 
            np.log1p(cluster_books['num_rating']) * 0.3
        )
        
        recommendations = cluster_books.nlargest(n, 'score')[
            ['title', 'author', 'avg_rating', 'num_rating', 'category']
        ]
        
        return recommendations.reset_index(drop=True)
    
    def recommend_by_cluster(self, cluster_id: int, n: int = 10, 
                           min_rating: float = 4.0) -> pd.DataFrame:
        """
        Get top books from a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            n: Number of recommendations to return
            min_rating: Minimum average rating threshold
            
        Returns:
            DataFrame with recommended books
        """
        cluster_books = self.df[
            (self.df['cluster'] == cluster_id) & 
            (self.df['avg_rating'] >= min_rating)
        ].copy()
        
        cluster_books['score'] = (
            cluster_books['avg_rating'] * 0.7 + 
            np.log1p(cluster_books['num_rating']) * 0.3
        )
        
        recommendations = cluster_books.nlargest(n, 'score')[
            ['title', 'author', 'avg_rating', 'num_rating', 'category']
        ]
        
        return recommendations.reset_index(drop=True)
    
    def get_cluster_summary(self, cluster_id: int) -> dict:
        """
        Get summary statistics for a cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Dictionary with cluster statistics
        """
        cluster_books = self.df[self.df['cluster'] == cluster_id]
        
        summary = {
            'cluster_id': cluster_id,
            'category': cluster_books['category'].iloc[0] if 'category' in cluster_books.columns else f"Cluster {cluster_id}",
            'num_books': len(cluster_books),
            'avg_rating_mean': cluster_books['avg_rating'].mean(),
            'avg_rating_std': cluster_books['avg_rating'].std(),
            'popularity_mean': cluster_books['num_rating'].mean(),
            'popularity_median': cluster_books['num_rating'].median(),
            'top_authors': cluster_books['author'].value_counts().head(5).to_dict()
        }
        
        return summary
    
    def search_books(self, query: str, n: int = 10) -> pd.DataFrame:
        """
        Search for books by title or author.
        
        Args:
            query: Search query
            n: Maximum number of results to return
            
        Returns:
            DataFrame with matching books
        """
        # Search in both title and author
        mask = (
            self.df['title'].str.contains(query, case=False, na=False) |
            self.df['author'].str.contains(query, case=False, na=False)
        )
        
        results = self.df[mask].nlargest(n, 'num_rating')[
            ['title', 'author', 'avg_rating', 'num_rating', 'category']
        ]
        
        return results.reset_index(drop=True)
    
    def get_all_categories(self) -> List[str]:
        """
        Get list of all unique categories.
        
        Returns:
            List of category names
        """
        if 'category' in self.df.columns:
            return sorted(self.df['category'].unique().tolist())
        else:
            return [f"Cluster {i}" for i in sorted(self.df['cluster'].unique())]


def print_recommendations(recommendations: pd.DataFrame, title: str = "Recommendations"):
    """
    Print recommendations in a formatted way.
    
    Args:
        recommendations: DataFrame with recommendations
        title: Title for the output
    """
    if len(recommendations) == 0:
        print("No recommendations found.")
        return
    
    print(f"\n{'='*80}")
    print(f"📚 {title}")
    print('='*80)
    
    for idx, row in recommendations.iterrows():
        print(f"\n{idx + 1}. {row['title']}")
        print(f"   By: {row['author']}")
        print(f"   ⭐ Rating: {row['avg_rating']:.2f} | 📊 {row['num_rating']:,} ratings")
        if 'category' in row:
            print(f"   🏷️  Category: {row['category']}")
        print()


if __name__ == "__main__":
    # Example usage
    recommender = BookRecommender()
    
    # Test recommendation by title
    print("Testing recommendations for 'Harry Potter'...")
    recs = recommender.recommend_by_title("Harry Potter", n=5)
    print_recommendations(recs, "Books Similar to Harry Potter")
    
    # Test recommendation by author
    print("\nTesting recommendations for author 'Stephen King'...")
    recs = recommender.recommend_by_author("Stephen King", n=5)
    print_recommendations(recs, "Books Similar to Stephen King's Style")
    
    # Test search
    print("\nTesting search for 'hunger'...")
    results = recommender.search_books("hunger", n=5)
    print_recommendations(results, "Search Results for 'hunger'")
    
    # Get cluster summary
    print("\nCluster 2 Summary:")
    summary = recommender.get_cluster_summary(2)
    for key, value in summary.items():
        print(f"{key}: {value}")
