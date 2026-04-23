"""
Collaborative Filtering Product Recommendation System

This module implements both user-based and item-based collaborative filtering
for recommending products based on user ratings and behavior.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')


class CollaborativeFilteringRecommender:
    """
    A collaborative filtering recommendation system that can work in two modes:
    1. User-Based: Find similar users and recommend products they liked
    2. Item-Based: Find similar products based on user ratings
    """
    
    def __init__(self, data_path: str = None, dataframe: pd.DataFrame = None):
        """
        Initialize the recommendation system.
        
        Args:
            data_path: Path to CSV file containing product and rating data
            dataframe: Alternatively, pass a pandas DataFrame directly
        """
        if data_path:
            self.df = pd.read_csv(data_path)
        elif dataframe is not None:
            self.df = dataframe
        else:
            raise ValueError("Either data_path or dataframe must be provided")
        
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare the data by creating user-item rating matrix."""
        # Create user-item rating matrix
        self.user_item_matrix = self.df.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            aggfunc='mean'
        )
        
        # Fill NaN values with 0 (no rating)
        self.user_item_matrix = self.user_item_matrix.fillna(0)
        
        print(f"User-Item Matrix Shape: {self.user_item_matrix.shape}")
        print(f"Matrix Sparsity: {(self.user_item_matrix == 0).sum().sum() / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]) * 100:.2f}%")
    
    def compute_user_similarity(self, similarity_metric='cosine'):
        """
        Compute similarity between users.
        
        Args:
            similarity_metric: 'cosine' or 'euclidean'
        
        Returns:
            Similarity matrix between users
        """
        if similarity_metric == 'cosine':
            self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        elif similarity_metric == 'euclidean':
            # Convert euclidean distance to similarity (inverse)
            distances = np.zeros((self.user_item_matrix.shape[0], self.user_item_matrix.shape[0]))
            for i in range(self.user_item_matrix.shape[0]):
                for j in range(self.user_item_matrix.shape[0]):
                    distances[i, j] = euclidean(
                        self.user_item_matrix.iloc[i].values,
                        self.user_item_matrix.iloc[j].values
                    )
            # Convert distance to similarity
            self.user_similarity_matrix = 1 / (1 + distances)
        
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        return self.user_similarity_matrix
    
    def compute_item_similarity(self, similarity_metric='cosine'):
        """
        Compute similarity between items.
        
        Args:
            similarity_metric: 'cosine' or 'euclidean'
        
        Returns:
            Similarity matrix between items
        """
        if similarity_metric == 'cosine':
            self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        elif similarity_metric == 'euclidean':
            distances = np.zeros((self.user_item_matrix.shape[1], self.user_item_matrix.shape[1]))
            for i in range(self.user_item_matrix.shape[1]):
                for j in range(self.user_item_matrix.shape[1]):
                    distances[i, j] = euclidean(
                        self.user_item_matrix.iloc[:, i].values,
                        self.user_item_matrix.iloc[:, j].values
                    )
            self.item_similarity_matrix = 1 / (1 + distances)
        
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        return self.item_similarity_matrix
    
    def recommend_user_based(self, user_id: str, n_recommendations: int = 5, 
                             n_similar_users: int = 5):
        """
        Recommend products for a user using User-Based Collaborative Filtering.
        
        Steps:
        1. Find the K most similar users
        2. Get products rated highly by similar users
        3. Filter out products already rated by the target user
        
        Args:
            user_id: The user to make recommendations for
            n_recommendations: Number of products to recommend
            n_similar_users: Number of similar users to consider
        
        Returns:
            DataFrame with recommended products and predicted ratings
        """
        if self.user_similarity_matrix is None:
            self.compute_user_similarity()
        
        if user_id not in self.user_similarity_matrix.index:
            return pd.DataFrame(columns=['product_id', 'predicted_rating'])
        
        # Get similar users
        similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)[1:n_similar_users+1]
        
        # Get products rated by similar users
        similar_users_ratings = self.user_item_matrix.loc[similar_users.index]
        
        # Weighted average of ratings from similar users
        weighted_ratings = similar_users_ratings.T.dot(similar_users.values) / similar_users.sum()
        
        # Products already rated by the user
        user_rated = self.user_item_matrix.loc[user_id]
        user_rated_products = user_rated[user_rated > 0].index
        
        # Exclude products already rated
        recommendations = weighted_ratings.drop(user_rated_products, errors='ignore')
        recommendations = recommendations.sort_values(ascending=False)[:n_recommendations]
        
        return pd.DataFrame({
            'product_id': recommendations.index,
            'predicted_rating': recommendations.values
        })
    
    def recommend_item_based(self, user_id: str, n_recommendations: int = 5):
        """
        Recommend products for a user using Item-Based Collaborative Filtering.
        
        Steps:
        1. Get products rated by the user
        2. Find similar products to those rated highly
        3. Score products based on similarity to user-liked items
        
        Args:
            user_id: The user to make recommendations for
            n_recommendations: Number of products to recommend
        
        Returns:
            DataFrame with recommended products and predicted ratings
        """
        if self.item_similarity_matrix is None:
            self.compute_item_similarity()
        
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame(columns=['product_id', 'predicted_rating'])
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Get products the user has rated
        rated_products = user_ratings[user_ratings > 0]
        
        if len(rated_products) == 0:
            return pd.DataFrame(columns=['product_id', 'predicted_rating'])
        
        # Calculate scores for unrated products
        recommendation_scores = {}
        
        for product_id in self.user_item_matrix.columns:
            if product_id not in rated_products.index:
                # Get similarity scores for this product with all rated products
                similarities = self.item_similarity_matrix.loc[product_id, rated_products.index].values
                ratings = rated_products.values
                
                # Weighted average
                if similarities.sum() > 0:
                    score = (similarities * ratings).sum() / similarities.sum()
                    recommendation_scores[product_id] = score
        
        # Sort and return top recommendations
        recommendations = sorted(recommendation_scores.items(), 
                                key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return pd.DataFrame({
            'product_id': [r[0] for r in recommendations],
            'predicted_rating': [r[1] for r in recommendations]
        })
    
    def hybrid_recommendation(self, user_id: str, n_recommendations: int = 5,
                             user_weight: float = 0.5, item_weight: float = 0.5):
        """
        Combine user-based and item-based recommendations.
        
        Args:
            user_id: The user to make recommendations for
            n_recommendations: Number of products to recommend
            user_weight: Weight for user-based recommendations (0-1)
            item_weight: Weight for item-based recommendations (0-1)
        
        Returns:
            DataFrame with recommended products
        """
        user_recs = self.recommend_user_based(user_id, n_recommendations * 2)
        item_recs = self.recommend_item_based(user_id, n_recommendations * 2)
        
        # Combine and normalize scores
        all_scores = {}
        
        if not user_recs.empty:
            user_max = user_recs['predicted_rating'].max()
            for _, row in user_recs.iterrows():
                all_scores[row['product_id']] = user_weight * (row['predicted_rating'] / user_max if user_max > 0 else 0)
        
        if not item_recs.empty:
            item_max = item_recs['predicted_rating'].max()
            for _, row in item_recs.iterrows():
                if row['product_id'] in all_scores:
                    all_scores[row['product_id']] += item_weight * (row['predicted_rating'] / item_max if item_max > 0 else 0)
                else:
                    all_scores[row['product_id']] = item_weight * (row['predicted_rating'] / item_max if item_max > 0 else 0)
        
        # Sort and return
        recommendations = sorted(all_scores.items(), 
                                key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        return pd.DataFrame({
            'product_id': [r[0] for r in recommendations],
            'score': [r[1] for r in recommendations]
        })
    
    def get_product_details(self, product_id: str):
        """Get product details from the original dataframe."""
        product_data = self.df[self.df['product_id'] == product_id].iloc[0]
        return {
            'product_id': product_data['product_id'],
            'product_name': product_data['product_name'],
            'category': product_data['category'],
            'rating': product_data['rating'],
            'price': product_data['discounted_price'],
            'original_price': product_data['actual_price'],
            'rating_count': product_data['rating_count']
        }
    
    def get_user_recommendations_with_details(self, user_id: str, method: str = 'hybrid',
                                              n_recommendations: int = 5):
        """
        Get recommendations with full product details.
        
        Args:
            user_id: User ID
            method: 'user_based', 'item_based', or 'hybrid'
            n_recommendations: Number of recommendations
        
        Returns:
            DataFrame with recommendations and product details
        """
        if method == 'user_based':
            recs = self.recommend_user_based(user_id, n_recommendations)
        elif method == 'item_based':
            recs = self.recommend_item_based(user_id, n_recommendations)
        else:  # hybrid
            recs = self.hybrid_recommendation(user_id, n_recommendations)
        
        if recs.empty:
            return pd.DataFrame()
        
        # Enrich with product details
        details = []
        for product_id in recs['product_id']:
            try:
                details.append(self.get_product_details(product_id))
            except:
                pass
        
        return pd.DataFrame(details)


class CollaborativeFilteringEvaluator:
    """Evaluate the performance of collaborative filtering recommendations."""
    
    @staticmethod
    def calculate_precision_at_k(recommendations, relevant_items, k=5):
        """
        Calculate Precision@K
        
        Args:
            recommendations: List of recommended items
            relevant_items: List of relevant items for the user
            k: Number of top recommendations to consider
        
        Returns:
            Precision score
        """
        if len(recommendations) == 0:
            return 0.0
        
        rec_at_k = recommendations[:k]
        relevant_in_rec = len([r for r in rec_at_k if r in relevant_items])
        return relevant_in_rec / k
    
    @staticmethod
    def calculate_recall_at_k(recommendations, relevant_items, k=5):
        """
        Calculate Recall@K
        
        Args:
            recommendations: List of recommended items
            relevant_items: List of relevant items for the user
            k: Number of top recommendations to consider
        
        Returns:
            Recall score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        rec_at_k = recommendations[:k]
        relevant_in_rec = len([r for r in rec_at_k if r in relevant_items])
        return relevant_in_rec / len(relevant_items)
    
    @staticmethod
    def calculate_mrr(recommendations, relevant_items):
        """
        Calculate Mean Reciprocal Rank
        
        Args:
            recommendations: List of recommended items (ordered)
            relevant_items: List of relevant items
        
        Returns:
            MRR score
        """
        for i, rec in enumerate(recommendations, 1):
            if rec in relevant_items:
                return 1 / i
        return 0.0


if __name__ == "__main__":
    # Example usage
    print("Collaborative Filtering Recommendation System")
    print("=" * 50)
