"""
Streamlit Web Interface for Product Recommendation System
Clean, Aesthetic, and Simple Design

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from recommendation_system import CollaborativeFilteringRecommender, CollaborativeFilteringEvaluator
from data_preprocessing import prepare_data_for_recommendation

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="Product Recommender",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling with transparent top nav
st.markdown("""
    <style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        padding: 2rem;
    }
    
    .top-nav {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 2rem;
        padding: 15px 0;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .nav-button {
        padding: 10px 20px;
        background: rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(102, 126, 234, 0.5);
        border-radius: 8px;
        color: #667eea;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
    }
    
    .nav-button:hover {
        background: rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.8);
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
    
    h1 {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .product-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home"

# ============================================================================
# HEADER & TOP NAVIGATION
# ============================================================================

st.markdown('<h1>🎁 Smart Product Recommender</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover products you\'ll love based on your preferences</p>', unsafe_allow_html=True)

# Top navigation buttons
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_page = "🏠 Home"

with col2:
    if st.button("🔍 Explore", use_container_width=True):
        st.session_state.current_page = "🔍 Explore"

with col3:
    if st.button("🎯 Recommendations", use_container_width=True):
        st.session_state.current_page = "🎯 Recommendations"

with col4:
    if st.button("📊 Analytics", use_container_width=True):
        st.session_state.current_page = "📊 Analytics"

with col5:
    if st.button("💰 Prices", use_container_width=True):
        st.session_state.current_page = "💰 Prices"

with col6:
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.current_page = "ℹ️ About"

st.markdown("---")

# ============================================================================
# PAGE ROUTING
# ============================================================================

page = st.session_state.current_page

# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ### Welcome! 👋
        
        Get personalized product recommendations using **Collaborative Filtering** - 
        a smart algorithm that learns from user preferences.
        
        **How it works:**
        - Analyzes how users rate products
        - Finds similar users and products
        - Recommends items you'll love
        """)
    
    with col2:
        if st.session_state.data is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="stat-box"><div class="stat-value">' + 
                           str(st.session_state.data['user_id'].nunique()) + 
                           '</div><div class="stat-label">Users</div></div>', 
                           unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="stat-box"><div class="stat-value">' + 
                           str(st.session_state.data['product_id'].nunique()) + 
                           '</div><div class="stat-label">Products</div></div>', 
                           unsafe_allow_html=True)
        else:
            st.info("📁 Load data to see statistics")
    
    st.markdown("---")
    
    # Additional stats section
    if st.session_state.data is not None:
        df_stats = st.session_state.data.copy()
        df_stats['discounted_price'] = df_stats['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
        
        st.markdown("### 📊 Quick Stats")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("⭐ Avg Rating", f"{df_stats['rating'].mean():.2f}")
        
        with col2:
            st.metric("💬 Total Reviews", f"{len(df_stats):,}")
        
        with col3:
            st.metric("🏷️ Categories", f"{df_stats['category'].nunique()}")
        
        with col4:
            st.metric("💰 Avg Price", f"₹{df_stats['discounted_price'].mean():.0f}")
        
        st.markdown("---")
    
    # Load data section
    st.markdown("### 📥 Setup")
    
    use_sparse_filter = st.checkbox(
        "Filter sparse users/products",
        value=False,
        help="Keep this off to load more users and products into the app."
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📂 Load Data", use_container_width=True, key="load_data"):
            with st.spinner("Loading Amazon dataset..."):
                try:
                    st.session_state.data = prepare_data_for_recommendation(
                        'Data/amazon.csv',
                        filter_sparse=use_sparse_filter
                    )
                    st.success("✅ Data loaded!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if st.session_state.data is not None:
            if st.button("⚙️ Initialize", use_container_width=True, key="init_rec"):
                with st.spinner("Computing similarities..."):
                    try:
                        st.session_state.recommender = CollaborativeFilteringRecommender(
                            dataframe=st.session_state.data
                        )
                        st.session_state.recommender.compute_user_similarity()
                        st.session_state.recommender.compute_item_similarity()
                        st.session_state.initialized = True
                        st.success("✅ Ready!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col3:
        if st.session_state.initialized:
            st.success("✅ System Ready")
        else:
            st.info("⏳ Complete setup")

# ============================================================================
# PAGE: EXPLORE
# ============================================================================

elif page == "🔍 Explore":
    if st.session_state.data is not None:
        df = st.session_state.data.copy()
        
        # Clean price data
        df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
        
        st.markdown("### 🔍 Explore Products")
        
        col1, col2 = st.columns(2)
        
        with col1:
            categories = sorted(df['category'].unique().tolist())
            selected_category = st.selectbox("Filter by Category:", ["All"] + categories)
        
        with col2:
            min_rating = st.slider("Minimum Rating:", 0.0, 5.0, 3.0)
        
        filtered_df = df.copy()
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
        
        st.info(f"📊 Showing {len(filtered_df)} products")
        st.markdown("---")
        
        for idx, row in filtered_df.head(20).iterrows():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**{row['product_name'][:70]}**")
                st.caption(row['category'][:60])
            
            with col2:
                st.metric("⭐ Rating", f"{row['rating']:.1f}")
            
            with col3:
                st.metric("💰 Price", f"₹{row['discounted_price']:.0f}")
            
            st.markdown("---")
    else:
        st.warning("⚠️ Please load data on Home page first")

# ============================================================================
# PAGE: PRICES
# ============================================================================

elif page == "💰 Prices":
    if st.session_state.data is not None:
        df = st.session_state.data.copy()
        
        # Clean price data
        df['discounted_price'] = pd.to_numeric(
            df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', ''),
            errors='coerce'
        ).fillna(0)
        df['discount_percentage'] = pd.to_numeric(
            df['discount_percentage'].astype(str).str.replace('%', ''),
            errors='coerce'
        ).fillna(0)
        df['rating_count'] = pd.to_numeric(
            df['rating_count'].astype(str).str.replace(',', ''),
            errors='coerce'
        ).fillna(0)
        
        st.markdown("### 💰 Price Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💵 Avg Price", f"₹{df['discounted_price'].mean():.0f}")
        col2.metric("📈 Max Price", f"₹{df['discounted_price'].max():.0f}")
        col3.metric("📉 Min Price", f"₹{df['discounted_price'].min():.0f}")
        col4.metric("💸 Median Price", f"₹{df['discounted_price'].median():.0f}")
        
        st.markdown("---")
        
        st.markdown("### 📊 Price Distribution")
        fig = px.histogram(df, x='discounted_price', nbins=50, 
                          title="Product Price Distribution",
                          labels={'discounted_price': 'Price (₹)'})
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🎉 Discount Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='discount_percentage', nbins=30,
                             title="Discount Percentage Distribution",
                             labels={'discount_percentage': 'Discount %'})
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_discount = df.groupby('category')['discount_percentage'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=avg_discount.values, y=avg_discount.index,
                         title="Avg Discount by Category",
                         labels={'x': 'Average Discount %', 'y': 'Category'})
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 📈 Price vs Rating")
        sample_size = min(500, len(df))
        fig = px.scatter(df.sample(sample_size), x='discounted_price', y='rating',
                        color='rating', size='rating_count',
                        title="Product Price vs Rating",
                        labels={'discounted_price': 'Price (₹)', 'rating': 'Rating'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 💎 Most Expensive Products")
        expensive = df.nlargest(10, 'discounted_price')[['product_name', 'discounted_price', 'rating', 'category']]
        
        for idx, (_, row) in enumerate(expensive.iterrows(), 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{idx}. {row['product_name'][:60]}**")
                st.caption(f"{row['category'][:50]}")
            with col2:
                st.markdown(f"**₹{row['discounted_price']:.0f}** | ⭐{row['rating']:.1f}")
            st.markdown("---")
        
    else:
        st.warning("⚠️ Please load data on Home page first")

# ============================================================================
# PAGE: RECOMMENDATIONS
# ============================================================================

elif page == "🎯 Recommendations":
    if not st.session_state.initialized or st.session_state.recommender is None:
        st.warning("⚠️ Please complete setup on Home page first")
    else:
        # User selection
        users = st.session_state.data['user_id'].unique().tolist()[:100]
        selected_user = st.selectbox("Select a user:", users)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            method = st.radio("Method:", ["User-Based", "Item-Based", "Hybrid"], horizontal=False)
        
        with col2:
            n_recs = st.slider("Number of recommendations:", 1, 10, 5)
        
        with col3:
            if method == "Hybrid":
                user_weight = st.slider("User weight:", 0.0, 1.0, 0.5)
                item_weight = 1.0 - user_weight
            else:
                user_weight = 1.0 if method == "User-Based" else 0.0
                item_weight = 1.0 - user_weight
        
        # Get recommendations
        if st.button("🚀 Get Recommendations", use_container_width=True):
            with st.spinner("Finding perfect products for you..."):
                try:
                    if method == "User-Based":
                        recs = st.session_state.recommender.recommend_user_based(
                            selected_user, n_recommendations=n_recs
                        )
                    elif method == "Item-Based":
                        recs = st.session_state.recommender.recommend_item_based(
                            selected_user, n_recommendations=n_recs
                        )
                    else:
                        recs = st.session_state.recommender.hybrid_recommendation(
                            selected_user, n_recommendations=n_recs,
                            user_weight=user_weight, item_weight=item_weight
                        )
                    
                    # Display results
                    st.markdown(f"### 🎁 Top {n_recs} Recommendations for {selected_user}")
                    
                    if recs.empty:
                        st.info("No recommendations available for this user")
                    else:
                        for idx, row in recs.iterrows():
                            try:
                                product = st.session_state.recommender.get_product_details(
                                    row['product_id']
                                )
                                
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"""
                                    **{idx+1}. {product['product_name'][:60]}**
                                    
                                    📦 {product['category'][:40]} | ⭐ {product['rating']}/5.0 | 💰 {product['price']}
                                    """)
                                
                                with col2:
                                    score = row.get('predicted_rating', row.get('score', 0))
                                    st.metric("Score", f"{score:.2f}")
                                
                                st.markdown("---")
                            except:
                                pass
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

elif page == "📊 Analytics":
    if not st.session_state.initialized or st.session_state.data is None:
        st.warning("⚠️ Please load data on Home page first")
    else:
        df = st.session_state.data.copy()
        
        # Clean data types
        df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
        df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(float)
        
        # Dataset stats
        st.markdown("### 📈 Dataset Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Users", f"{df['user_id'].nunique():,}")
        col2.metric("📦 Products", f"{df['product_id'].nunique():,}")
        col3.metric("⭐ Avg Rating", f"{df['rating'].mean():.2f}")
        col4.metric("💬 Total Ratings", f"{len(df):,}")
        
        st.markdown("---")
        
        # Rating distribution
        st.markdown("### ⭐ Rating Distribution")
        
        rating_dist = df['rating'].value_counts().sort_index()
        fig = px.bar(x=rating_dist.index, y=rating_dist.values,
                    labels={'x': 'Rating', 'y': 'Count'},
                    color=rating_dist.values,
                    color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top categories
        st.markdown("### 📂 Top Categories")
        
        category_counts = df['category'].value_counts().head(10)
        fig = px.bar(x=category_counts.values, y=category_counts.index,
                orientation='h',
                labels={'x': 'Count', 'y': 'Category'},
                color=category_counts.values,
                color_continuous_scale='Blues')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Top products
        st.markdown("### 🏆 Top Rated Products")
        
        top_products = df.groupby('product_id').agg({
            'product_name': 'first',
            'rating': 'mean',
            'rating_count': 'first'
        }).sort_values('rating', ascending=False).head(10)
        
        for idx, (pid, row) in enumerate(top_products.iterrows(), 1):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**{idx}. {row['product_name'][:50]}**")
            with col2:
                st.metric("Rating", f"{row['rating']:.1f}")
        
        st.markdown("---")
        
        # Rating count distribution
        st.markdown("### 📊 Rating Count Distribution")
        
        fig = px.box(df, y='rating_count', title="Distribution of Product Review Counts",
                    labels={'rating_count': 'Number of Reviews'})
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Category vs Average Rating
        st.markdown("### ⭐ Average Rating by Category")
        
        cat_rating = df.groupby('category')['rating'].mean().sort_values(ascending=False).head(15)
        fig = px.bar(x=cat_rating.values, y=cat_rating.index,
                orientation='h',
                title="Top 15 Categories by Average Rating",
                labels={'x': 'Average Rating', 'y': 'Category'},
                color=cat_rating.values,
                color_continuous_scale='Greens')
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Heatmap style: Products per category and rating
        st.markdown("### 📈 Rating Distribution Across Top Categories")
        
        top_cats = df['category'].value_counts().head(5).index.tolist()
        category_rating = df[df['category'].isin(top_cats)].groupby(['category', 'rating']).size().reset_index(name='count')
        
        fig = px.bar(category_rating, x='rating', y='count', color='category',
                    title="Rating Distribution in Top 5 Categories",
                    barmode='group',
                    labels={'count': 'Number of Products', 'rating': 'Rating'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Most reviewed products
        st.markdown("### 🔥 Most Reviewed Products")
        
        most_reviewed = df.nlargest(10, 'rating_count')[['product_name', 'rating_count', 'rating', 'category']]
        
        for idx, (_, row) in enumerate(most_reviewed.iterrows(), 1):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{idx}. {row['product_name'][:55]}**")
            with col2:
                st.metric("Reviews", f"{int(row['rating_count']):,}")
            with col3:
                st.metric("Rating", f"{row['rating']:.1f}")
            st.markdown("---")

# ============================================================================
# PAGE: ABOUT
# ============================================================================

elif page == "ℹ️ About":
    st.markdown("### 🎁 About Smart Product Recommender")
    
    st.info("""
    **Smart Product Recommender** uses advanced algorithms to suggest products 
    you'll love based on user behavior patterns and preferences.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 How It Works
        
        **Collaborative Filtering** analyzes:
        - How users rate products
        - Similar user preferences
        - Product similarity patterns
        
        Then recommends products based on what similar users liked!
        """)
    
    with col2:
        st.markdown("""
        #### 🔧 Three Methods
        
        1. **User-Based** - Find similar users
        2. **Item-Based** - Find similar products
        3. **Hybrid** - Best of both
        
        Choose what works best for you!
        """)
    
    st.markdown("---")
    
    st.markdown("""
    #### � Pages & Features
    
    🏠 **Home** - Load data and initialize the recommender system
    
    🔍 **Explore** - Browse products by category and rating filters
    
    🎯 **Recommendations** - Get personalized product suggestions using three different methods
    
    📊 **Analytics** - View comprehensive data insights:
    - Rating distributions and statistics
    - Category analysis and trends
    - Top-rated and most-reviewed products
    - Rating patterns across categories
    
    💰 **Prices** - Analyze pricing data:
    - Price distribution and statistics
    - Discount analysis by category
    - Price vs Rating correlation
    - Most expensive products
    
    ℹ️ **About** - Learn about the system and features
    """)
    
    st.markdown("---")
    
    st.markdown("""
    #### 📊 Evaluation Metrics
    
    - **Precision@K** - Accuracy of recommendations
    - **Recall@K** - Coverage of good products
    - **F1-Score** - Overall quality
    
    #### 🚀 Technology Stack
    
    - **Backend**: Python 3.7+ • Pandas • NumPy • Scikit-learn
    - **Frontend**: Streamlit • Plotly
    - **Algorithms**: Collaborative Filtering (User-Based, Item-Based, Hybrid)
    - **Data**: Amazon Product Reviews & Ratings
    
    #### ✨ Key Features
    
    ✅ Fast product recommendations  
    ✅ Beautiful interactive visualizations  
    ✅ Comprehensive data analytics  
    ✅ Multiple recommendation methods  
    ✅ Price & discount analysis  
    ✅ Product exploration & filtering  
    ✅ Simple, clean interface  
    
    #### 📈 Data Insights
    
    - Analyze 1000+ products across multiple categories
    - 10,000+ user reviews and ratings
    - Interactive charts and statistics
    - Real-time data filtering and analysis
    
    Made with ❤️ for a better shopping experience!
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #999; padding: 20px;'>
    <p>🎁 Smart Product Recommender v2.0 | Powered by Collaborative Filtering</p>
    </div>
""", unsafe_allow_html=True)
