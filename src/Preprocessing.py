"""
Data preprocessing module for book clustering project.
Handles data cleaning, feature extraction, and transformation.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Goodreads dataset.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with raw book data
    """
    df = pd.read_csv(filepath)
    return df


def extract_rating_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract average rating and number of ratings from rating_info column.
    
    Args:
        df: DataFrame with 'rating_info' column
        
    Returns:
        DataFrame with extracted 'avg_rating' and 'num_rating' columns
    """
    df = df.copy()
    
    # Extract average rating (e.g., "4.35 avg rating")
    df["avg_rating"] = df["rating_info"].str.extract(r"([0-9]\.\d+)").astype(float)
    
    # Extract number of ratings (e.g., "— 9,682,830 ratings")
    df["num_rating"] = (
        df["rating_info"]
        .str.extract(r"— ([\d,]+) ratings")[0]
        .str.replace(",", "", regex=True)
        .astype(int)
    )
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and removing duplicates.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove duplicates
    initial_shape = df.shape[0]
    df = df.drop_duplicates(subset=['title', 'author'], keep='first')
    duplicates_removed = initial_shape - df.shape[0]
    print(f"Removed {duplicates_removed} duplicate books")
    
    # Handle missing values
    df = df.dropna(subset=['title', 'author', 'avg_rating', 'num_rating'])
    
    # Remove outliers (books with very few ratings - likely noise)
    df = df[df['num_rating'] >= 100]
    
    print(f"Final dataset size: {df.shape[0]} books")
    
    return df.reset_index(drop=True)


def create_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined text features from title and author.
    
    Args:
        df: DataFrame with 'title' and 'author' columns
        
    Returns:
        DataFrame with added 'text' column
    """
    df = df.copy()
    df["text"] = df["title"] + " " + df["author"]
    return df


def preprocess_pipeline(input_filepath: str, output_filepath: str) -> pd.DataFrame:
    """
    Complete preprocessing pipeline from raw data to clean dataset.
    
    Args:
        input_filepath: Path to raw CSV file
        output_filepath: Path to save cleaned CSV file
        
    Returns:
        Preprocessed DataFrame
    """
    print("Loading data...")
    df = load_data(input_filepath)
    
    print("Extracting rating information...")
    df = extract_rating_info(df)
    
    print("Cleaning data...")
    df = clean_data(df)
    
    print("Creating text features...")
    df = create_text_features(df)
    
    # Drop original rating_info column
    if 'rating_info' in df.columns:
        df = df.drop(columns=['rating_info'])
    
    print(f"Saving cleaned data to {output_filepath}")
    df.to_csv(output_filepath, index=False)
    
    return df


if __name__ == "__main__":
    # Example usage
    df = preprocess_pipeline(
        "../data/goodreads_books.csv",
        "../data/goodreads_books_cleaned.csv"
    )
    print("\nData preview:")
    print(df.head())
    print("\nData info:")
    print(df.info())
