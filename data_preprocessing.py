"""
Data Preprocessing Script for Product Recommendation System

This module contains functions to load, clean, and preprocess the Amazon product data.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handle data loading and preprocessing for recommendation system."""
    
    @staticmethod
    def load_data(csv_path: str) -> pd.DataFrame:
        """
        Load CSV data.
        
        Args:
            csv_path: Path to the CSV file
        
        Returns:
            DataFrame with loaded data
        """
        df = pd.read_csv(csv_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    @staticmethod
    def clean_price_column(price_str):
        """Convert price string to float."""
        if pd.isna(price_str):
            return 0.0
        if isinstance(price_str, (int, float)):
            return float(price_str)
        # Remove currency symbols and commas
        cleaned = str(price_str).replace('₹', '').replace(',', '').strip()
        try:
            return float(cleaned)
        except:
            return 0.0
    
    @staticmethod
    def clean_rating_count(count_str):
        """Convert rating count string to integer."""
        if pd.isna(count_str):
            return 0
        if isinstance(count_str, (int, float)):
            return int(count_str)
        # Remove commas
        cleaned = str(count_str).replace(',', '').strip()
        try:
            return int(cleaned)
        except:
            return 0
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['product_id', 'user_id'])
        print(f"Removed duplicates: {initial_rows - len(df_clean)} rows")
        
        # Clean price columns
        df_clean['discounted_price_numeric'] = df_clean['discounted_price'].apply(
            DataPreprocessor.clean_price_column
        )
        df_clean['actual_price_numeric'] = df_clean['actual_price'].apply(
            DataPreprocessor.clean_price_column
        )
        
        # Clean rating count
        df_clean['rating_count_numeric'] = df_clean['rating_count'].apply(
            DataPreprocessor.clean_rating_count
        )
        
        # Ensure rating is numeric and in valid range
        df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')
        df_clean = df_clean[(df_clean['rating'] >= 0) & (df_clean['rating'] <= 5)]
        
        # Handle missing values
        df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].median())
        df_clean['user_id'] = df_clean['user_id'].fillna('unknown_user')
        df_clean['product_id'] = df_clean['product_id'].fillna('unknown_product')
        
        print(f"Data cleaned: {len(df_clean)} rows remaining")
        print(f"Rating range: {df_clean['rating'].min()} - {df_clean['rating'].max()}")
        print(f"Rating distribution:\n{df_clean['rating'].value_counts().sort_index()}")
        
        return df_clean
    
    @staticmethod
    def get_user_statistics(df: pd.DataFrame) -> dict:
        """Get statistics about users in the dataset."""
        return {
            'total_users': df['user_id'].nunique(),
            'total_products': df['product_id'].nunique(),
            'total_ratings': len(df),
            'avg_ratings_per_user': df.groupby('user_id').size().mean(),
            'avg_ratings_per_product': df.groupby('product_id').size().mean(),
            'avg_rating_value': df['rating'].mean(),
            'sparsity': 1 - (len(df) / (df['user_id'].nunique() * df['product_id'].nunique()))
        }
    
    @staticmethod
    def filter_data(df: pd.DataFrame, min_user_ratings: int = 2, 
                    min_product_ratings: int = 2) -> pd.DataFrame:
        """
        Filter out users and products with too few ratings.
        
        Args:
            df: Input DataFrame
            min_user_ratings: Minimum ratings per user
            min_product_ratings: Minimum ratings per product
        
        Returns:
            Filtered DataFrame
        """
        df_filtered = df.copy()
        
        # Filter users
        user_counts = df_filtered['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df_filtered = df_filtered[df_filtered['user_id'].isin(valid_users)]
        
        # Filter products
        product_counts = df_filtered['product_id'].value_counts()
        valid_products = product_counts[product_counts >= min_product_ratings].index
        df_filtered = df_filtered[df_filtered['product_id'].isin(valid_products)]
        
        print(f"After filtering - Users: {df_filtered['user_id'].nunique()}, "
              f"Products: {df_filtered['product_id'].nunique()}, "
              f"Ratings: {len(df_filtered)}")
        
        return df_filtered
    
    @staticmethod
    def get_top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get top rated products."""
        return df.groupby('product_id').agg({
            'product_name': 'first',
            'rating': ['mean', 'count'],
            'discounted_price_numeric': 'first',
            'category': 'first'
        }).round(2).sort_values(('rating', 'mean'), ascending=False).head(n)
    
    @staticmethod
    def get_product_category_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Get statistics by product category."""
        return df.groupby('category').agg({
            'product_id': 'nunique',
            'user_id': 'nunique',
            'rating': ['mean', 'count']
        }).round(2).sort_values(('rating', 'mean'), ascending=False)


def prepare_data_for_recommendation(csv_path: str, filter_sparse: bool = False):
    """
    Complete pipeline to prepare data for recommendation system.
    
    Args:
        csv_path: Path to CSV file
        filter_sparse: Whether to filter sparse data
    
    Returns:
        Cleaned and prepared DataFrame
    """
    # Load data
    df = DataPreprocessor.load_data(csv_path)
    
    # Preprocess
    df = DataPreprocessor.preprocess_data(df)
    
    # Print statistics
    print("\nDataset Statistics:")
    stats = DataPreprocessor.get_user_statistics(df)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Filter sparse data
    if filter_sparse:
        print("\nFiltering sparse data...")
        df = DataPreprocessor.filter_data(df, min_user_ratings=2, min_product_ratings=2)
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        df = prepare_data_for_recommendation(csv_path)
    else:
        print("Usage: python data_preprocessing.py <path_to_csv>")
