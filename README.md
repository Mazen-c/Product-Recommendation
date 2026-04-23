# Product Recommendation System - Collaborative Filtering

A comprehensive **Collaborative Filtering-based Product Recommendation Engine** built for Amazon product data. This system implements both user-based and item-based collaborative filtering to provide intelligent product recommendations.

## 📋 Project Structure

```
Product Recommendation/
├── Data/
│   └── amazon.csv                      # Amazon product dataset with ratings
│
├── Core Recommendation Engine
│   ├── recommendation_system.py         # Main CF algorithms
│   └── data_preprocessing.py            # Data cleaning and preparation
│
├── User Interface
│   └── streamlit_app.py                 # Web interface (Streamlit)
│
├── Demo & Examples
│   └── Product_Recommendation_Demo.ipynb # Interactive Jupyter notebook
│
├── Documentation
│   ├── README.md                        # This file
│   ├── QUICKSTART.md                    # 2-minute setup guide
│   └── requirements.txt                 # Python dependencies
```

## 🎯 What is Collaborative Filtering?

Collaborative Filtering is a recommendation technique that:
- Analyzes user-item interactions (ratings, purchases, etc.)
- Finds patterns in how users rate products
- Recommends items based on similar users' preferences

### Two Main Approaches:

1. **User-Based Collaborative Filtering**
   - Find users similar to target user
   - Recommend products those similar users liked
   - Formula: Score = Weighted average of similar users' ratings

2. **Item-Based Collaborative Filtering**
   - Find items similar to products user liked
   - Recommend similar items
   - Formula: Score = Weighted average based on item similarity

## 📊 Dataset

The system works with Amazon product data containing:
- **product_id**: Unique product identifier
- **product_name**: Name of the product
- **rating**: User rating (1-5 stars)
- **user_id**: Unique user identifier
- **category**: Product category
- **discounted_price**: Current price
- **rating_count**: Number of ratings received

## 🚀 Quick Start

### 1. Installation

```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Using the Recommendation System

```python
from recommendation_system import CollaborativeFilteringRecommender
from data_preprocessing import prepare_data_for_recommendation

# Prepare data
df = prepare_data_for_recommendation('Data/amazon.csv')

# Initialize recommender
recommender = CollaborativeFilteringRecommender(dataframe=df)

# Get user-based recommendations
user_recs = recommender.recommend_user_based(
    user_id='some_user_id',
    n_recommendations=5,
    n_similar_users=10
)

# Get item-based recommendations
item_recs = recommender.recommend_item_based(
    user_id='some_user_id',
    n_recommendations=5
)

# Get hybrid recommendations (combines both)
hybrid_recs = recommender.hybrid_recommendation(
    user_id='some_user_id',
    n_recommendations=5
)

# Get detailed recommendations with product info
detailed_recs = recommender.get_user_recommendations_with_details(
    user_id='some_user_id',
    method='hybrid',
    n_recommendations=5
)
```

## 📖 Module Documentation

### `recommendation_system.py`

#### `CollaborativeFilteringRecommender` Class

Main recommendation engine with the following methods:

**Constructor:**
```python
recommender = CollaborativeFilteringRecommender(data_path=None, dataframe=None)
```

**Key Methods:**

- `compute_user_similarity(similarity_metric='cosine')`: Calculate user-user similarity
- `compute_item_similarity(similarity_metric='cosine')`: Calculate item-item similarity
- `recommend_user_based(user_id, n_recommendations=5, n_similar_users=5)`: User-based recommendations
- `recommend_item_based(user_id, n_recommendations=5)`: Item-based recommendations
- `hybrid_recommendation(user_id, n_recommendations=5, user_weight=0.5, item_weight=0.5)`: Combined approach
- `get_user_recommendations_with_details(user_id, method='hybrid', n_recommendations=5)`: Get detailed recommendations

#### `CollaborativeFilteringEvaluator` Class

Evaluation metrics for recommendation quality:

- `calculate_precision_at_k()`: Precision@K metric
- `calculate_recall_at_k()`: Recall@K metric
- `calculate_mrr()`: Mean Reciprocal Rank

### `data_preprocessing.py`

Data preparation utilities:

#### `DataPreprocessor` Class

**Key Methods:**
- `load_data(csv_path)`: Load CSV file
- `preprocess_data(df)`: Clean and normalize data
- `filter_data(df, min_user_ratings, min_product_ratings)`: Remove sparse data
- `get_user_statistics(df)`: Dataset statistics
- `get_top_products(df, n)`: Get top-rated products
- `get_product_category_stats(df)`: Category-wise statistics

**Function:**
```python
df_clean = prepare_data_for_recommendation('Data/amazon.csv', filter_sparse=True)
```

## 🔧 How It Works

### Step 1: Data Preparation
```python
from data_preprocessing import prepare_data_for_recommendation
df = prepare_data_for_recommendation('Data/amazon.csv')
```

### Step 2: Create User-Item Matrix
- Rows: Users
- Columns: Products  
- Values: Ratings (0 = no rating)

### Step 3: Calculate Similarities
- **Cosine Similarity**: Measures angle between rating vectors
- Formula: `similarity = (A · B) / (||A|| × ||B||)`

### Step 4: Generate Recommendations

**User-Based:**
1. Find K nearest users (highest similarity)
2. Look at products rated by similar users
3. Score products using weighted average
4. Exclude products already rated
5. Return top N products

**Item-Based:**
1. Get products rated by user
2. Find similar products
3. Score based on similarity to liked items
4. Exclude already-rated products
5. Return top N products

## 📈 Evaluation Metrics

### Precision@K
- Fraction of top-K recommendations that are relevant
- Formula: `Precision@K = (Relevant in Top-K) / K`

### Recall@K
- Fraction of all relevant items captured in top-K
- Formula: `Recall@K = (Relevant in Top-K) / (Total Relevant)`

### F1-Score
- Harmonic mean of Precision and Recall
- Formula: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

## ⚠️ Limitations & Challenges

### 1. **Data Sparsity**
- Most user-item pairs have no interaction
- Solution: Use hybrid approach, dimensionality reduction

### 2. **Cold Start Problem**
- New users: No rating history
- New items: No user feedback
- Solution: Use content-based features, popularity-based recommendations

### 3. **Scalability**
- Computing full similarity matrices is O(n²)
- Solution: Approximate nearest neighbors, distributed computing

### 4. **Popularity Bias**
- Popular items get recommended more often
- Solution: Re-ranking, diversity-aware recommendations

## 🎓 Example Usage

### Example 1: Get Recommendations for a User

```python
from recommendation_system import CollaborativeFilteringRecommender
from data_preprocessing import prepare_data_for_recommendation

# Setup
df = prepare_data_for_recommendation('Data/amazon.csv')
recommender = CollaborativeFilteringRecommender(dataframe=df)

# Get recommendations
user_id = df['user_id'].iloc[0]
recommendations = recommender.hybrid_recommendation(
    user_id=user_id,
    n_recommendations=5
)

# Display
for idx, row in recommendations.iterrows():
    product = recommender.get_product_details(row['product_id'])
    print(f"{idx+1}. {product['product_name']}")
    print(f"   Rating: {product['rating']}")
    print(f"   Price: {product['price']}")
```

### Example 2: Evaluate Recommendation Quality

```python
from recommendation_system import CollaborativeFilteringEvaluator

# Simulate recommendations and relevant items
recommendations = ['P1', 'P2', 'P3', 'P4', 'P5']
relevant_items = ['P1', 'P3', 'P7']

# Calculate metrics
precision = CollaborativeFilteringEvaluator.calculate_precision_at_k(
    recommendations, relevant_items, k=5
)
recall = CollaborativeFilteringEvaluator.calculate_recall_at_k(
    recommendations, relevant_items, k=5
)

print(f"Precision@5: {precision:.3f}")
print(f"Recall@5: {recall:.3f}")
```

## 📚 Advanced Topics

### Matrix Factorization
Instead of full similarity matrices, decompose into latent factors:
- SVD (Singular Value Decomposition)
- NMF (Non-negative Matrix Factorization)
- ALS (Alternating Least Squares)

### Deep Learning Approaches
- Neural Collaborative Filtering
- Autoencoders
- Embedding-based models

### Hybrid Systems
- Combine collaborative filtering with content-based
- Use metadata, categories, brands
- Incorporate temporal dynamics

## 🔄 Improving Recommendations

1. **Add More Features**
   - User demographics (age, location)
   - Product features (brand, color, size)
   - Contextual information (season, trends)

2. **Use Feedback Loops**
   - Implicit feedback (clicks, purchases)
   - Explicit feedback (ratings, reviews)
   - User interactions

3. **Diversify Results**
   - Include different categories
   - Balance popularity and novelty
   - Serendipitous recommendations

4. **Optimize for Business Goals**
   - Revenue maximization
   - User satisfaction
   - Engagement metrics

## 🌐 User Interface

This project includes a beautiful, simple **Streamlit Web Application** that makes it easy to get product recommendations.

### Running the App

```bash
# Make sure you have installed requirements
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

The app will open in your browser at **http://localhost:8501**

### Features

**🏠 Home Page**
- Load Amazon product dataset
- Initialize recommendation system
- View dataset statistics

**🎯 Recommendations**
- Select any user
- Choose recommendation method (User-Based, Item-Based, or Hybrid)
- Get personalized product recommendations
- Adjust number of recommendations (1-10)
- View product details, ratings, and prices

**📊 Analytics**
- Rating distribution charts
- Top product categories
- Top rated products
- Key statistics

**ℹ️ About**
- Learn how the system works
- Understand the algorithms
- View technology stack

---

## 📦 File Descriptions

| File | Purpose |
|------|---------|
| `recommendation_system.py` | Core engine with CF algorithms |
| `data_preprocessing.py` | Data cleaning and preparation |
| `streamlit_app.py` | Web interface (Streamlit) |
| `Product_Recommendation_Demo.ipynb` | Interactive Jupyter notebook |
| `requirements.txt` | Python dependencies |
| `Data/amazon.csv` | Product dataset |

---

## 📊 Support & Troubleshooting


### Common Issues

**Q: "No recommendations available"**
- A: User may have rated too few items or be completely new
- Solution: Use popularity-based fallback

**Q: "Recommendations are all popular items"**
- A: Data may be biased towards popular products
- Solution: Use re-ranking or diversity metrics

**Q: "System is slow"**
- A: Computing similarity matrices is expensive
- Solution: Use approximate methods or caching

## 🤝 Contributing

Feel free to extend this system with:
- More sophisticated algorithms
- Better evaluation metrics
- Visualization tools
- API deployment

## 📄 License

This project is provided as-is for educational and research purposes.

## 📞 Contact & References

For questions or improvements, refer to:
- Scikit-learn documentation
- Collaborative Filtering papers
- Recommendation systems literature

---

**Happy Recommending! 🚀**
