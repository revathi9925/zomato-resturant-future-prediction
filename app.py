import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Must be the first Streamlit command
st.set_page_config(
    page_title="Zomato Advanced Analytics", 
    layout="wide", 
    page_icon="🍽️",
    initial_sidebar_state="collapsed"
)

# Professional CSS with anchor link fix
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    /* Hide anchor links that appear as "Error" */
    a[href*="#"] { display: none !important; }
    .stApp a[href*="#"] { display: none !important; }
    .element-container a[href*="#"] { display: none !important; }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        text-align: center;
        color: white;
        transition: transform 0.3s;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .insight-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .restaurant-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 0.8rem 0;
        border: 1px solid #f0f0f0;
        transition: all 0.3s;
    }
    
    .restaurant-card:hover {
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transform: translateX(5px);
    }
    
    .badge {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-success { background: #4CAF50; color: white; }
    .badge-warning { background: #FFC107; color: black; }
    .badge-danger { background: #FF5252; color: white; }
    .badge-info { background: #667eea; color: white; }
</style>
""", unsafe_allow_html=True)

# Generate realistic restaurant data with caching
@st.cache_data(ttl=3600)
def generate_restaurant_data():
    np.random.seed(42)
    n = 500
    
    prefixes = ['The', 'Royal', 'Spicy', 'Golden', 'Urban', 'Rural', 'Modern', 'Classic', 'Elite', 'Grand']
    suffixes = ['Kitchen', 'Dining', 'Cafe', 'Restaurant', 'Bistro', 'House', 'Palace', 'Grill', 'Corner', 'Hub']
    cuisines = ['North Indian', 'Chinese', 'South Indian', 'Italian', 'Continental', 'Mexican', 'Thai', 'Japanese', 'Fast Food', 'Bakery']
    locations = ['Connaught Place', 'South Delhi', 'Gurgaon', 'Noida', 'Ghaziabad', 'CP', 'Karol Bagh', 'Lajpat Nagar']
    
    names = [f"{np.random.choice(prefixes)} {np.random.choice(suffixes)} {i}" for i in range(1, n+1)]
    
    # Generate correlated features
    ratings = np.random.normal(3.8, 0.6, n).clip(2.0, 5.0).round(1)
    votes = (ratings * 200 + np.random.normal(0, 100, n)).clip(10, 5000).astype(int)
    cost = (ratings * 200 + np.random.normal(400, 200, n)).clip(150, 3000).round(-1)
    
    data = pd.DataFrame({
        'name': names,
        'cuisine': np.random.choice(cuisines, n),
        'rating': ratings,
        'votes': votes,
        'cost_for_two': cost,
        'location': np.random.choice(locations, n),
        'online_order': np.random.choice(['Yes', 'No'], n, p=[0.75, 0.25]),
        'table_booking': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
    })
    
    # Calculate advanced metrics
    data['rating_score'] = data['rating'] / 5 * 40
    data['votes_score'] = np.log1p(data['votes']) / np.log1p(data['votes'].max()) * 30
    data['price_score'] = (1 - data['cost_for_two'] / data['cost_for_two'].max()) * 20
    data['online_score'] = (data['online_order'] == 'Yes').astype(int) * 10
    
    data['current_score'] = (data['rating_score'] + data['votes_score'] + 
                            data['price_score'] + data['online_score']).round(1)
    
    # Future score with some randomness
    growth_factor = np.random.normal(1.1, 0.1, n)
    data['future_score'] = (data['current_score'] * growth_factor).clip(0, 100).round(1)
    data['growth_rate'] = ((data['future_score'] - data['current_score']) / data['current_score'] * 100).round(1)
    
    # Growth category
    data['growth_category'] = pd.cut(data['growth_rate'], 
                                     bins=[-100, 0, 10, 20, 50, 100],
                                     labels=['Declining', 'Slow', 'Moderate', 'High', 'Rapid'])
    
    return data

# Load data
df = generate_restaurant_data()

# Header
st.markdown('<h1 class="main-title">🍽️ Zomato Restaurant Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Analytics • Future Predictions • Market Intelligence</p>', unsafe_allow_html=True)

# Top Metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Total Restaurants</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{df['rating'].mean():.2f}</div>
        <div class="metric-label">Avg Rating</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">₹{df['cost_for_two'].mean():.0f}</div>
        <div class="metric-label">Avg Cost</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    online_pct = (df['online_order'] == 'Yes').mean() * 100
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{online_pct:.1f}%</div>
        <div class="metric-label">Online Orders</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{df['future_score'].mean():.1f}</div>
        <div class="metric-label">Future Score</div>
    </div>
    """, unsafe_allow_html=True)

# Main Dashboard with Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance Analytics", 
    "🔮 Future Predictions", 
    "🤖 ML Segmentation",
    "🏆 Top Performers",
    "🎯 Smart Recommendations"
])

with tab1:
    st.markdown("### 📈 Restaurant Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rating Distribution
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Rating Distribution', 'Rating by Cuisine'))
        
        fig.add_trace(
            go.Histogram(x=df['rating'], nbinsx=20, marker_color='#667eea', name='Ratings'),
            row=1, col=1
        )
        
        # Rating by cuisine
        cuisine_rating = df.groupby('cuisine')['rating'].mean().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(x=cuisine_rating.values, y=cuisine_rating.index, orientation='h',
                  marker_color='#764ba2', showlegend=False),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Rating Analysis", showlegend=False)
        fig.update_xaxes(title_text="Rating", row=1, col=1)
        fig.update_xaxes(title_text="Average Rating", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 3D Scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=df['rating'],
            y=df['cost_for_two'],
            z=df['votes'],
            mode='markers',
            marker=dict(
                size=6,
                color=df['current_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Current Score")
            ),
            text=df['name'],
            hovertemplate='<b>%{text}</b><br>Rating: %{x}<br>Cost: ₹%{y}<br>Votes: %{z}<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D: Rating vs Price vs Votes',
            scene=dict(
                xaxis_title='Rating',
                yaxis_title='Cost for Two (₹)',
                zaxis_title='Votes',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.markdown("### 🔥 Feature Correlation Matrix")
    
    corr_cols = ['rating', 'votes', 'cost_for_two', 'current_score', 'future_score', 'growth_rate']
    corr_matrix = df[corr_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Between Key Metrics")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### 🔮 Future Popularity Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current vs Future Score
        fig = go.Figure()
        
        fig.add_trace(go.Violin(y=df['current_score'], name='Current Score', 
                                line_color='#667eea', side='positive'))
        fig.add_trace(go.Violin(y=df['future_score'], name='Future Score', 
                                line_color='#764ba2', side='positive'))
        
        fig.update_layout(title='Distribution: Current vs Future Scores', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Growth Rate by Category
        growth_by_cuisine = df.groupby('cuisine')['growth_rate'].mean().sort_values(ascending=True)
        
        fig = px.bar(x=growth_by_cuisine.values, y=growth_by_cuisine.index, orientation='h',
                    title='Expected Growth Rate by Cuisine',
                    color=growth_by_cuisine.values,
                    color_continuous_scale='RdYlGn',
                    labels={'x': 'Growth Rate (%)', 'y': ''})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Growth Category Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        growth_counts = df['growth_category'].value_counts()
        fig = px.pie(values=growth_counts.values, names=growth_counts.index,
                    title='Restaurants by Growth Category',
                    color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot with growth categories
        fig = px.scatter(df, x='current_score', y='future_score', 
                        color='growth_category',
                        size='votes', hover_name='name',
                        title='Current vs Future Score by Growth Category',
                        color_discrete_map={
                            'Rapid': '#4CAF50',
                            'High': '#8BC34A',
                            'Moderate': '#FFC107',
                            'Slow': '#FF9800',
                            'Declining': '#FF5252'
                        })
        fig.add_shape(type='line', x0=0, y0=0, x1=100, y1=100,
                     line=dict(color='gray', dash='dash'))
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 🤖 Machine Learning: Restaurant Segmentation")
    
    # Prepare data for clustering
    features = ['rating', 'votes', 'cost_for_two']
    X = df[features].copy()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means with fixed n_init
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 2D Cluster visualization
        fig = px.scatter(df, x='rating', y='cost_for_two', 
                        color=df['cluster'].astype(str),
                        size='votes', hover_name='name',
                        title='Customer Segments (K-Means Clustering)',
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        labels={'cluster': 'Segment', 'color': 'Segment'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cluster characteristics
        cluster_stats = df.groupby('cluster').agg({
            'rating': 'mean',
            'cost_for_two': 'mean',
            'votes': 'mean',
            'name': 'count',
            'future_score': 'mean'
        }).round(2)
        
        cluster_stats.columns = ['Avg Rating', 'Avg Price', 'Avg Votes', 'Count', 'Future Score']
        st.dataframe(cluster_stats, use_container_width=True)
    
    # Segment descriptions
    st.markdown("### 📊 Segment Analysis")
    
    for cluster in range(4):
        segment_data = df[df['cluster'] == cluster]
        
        avg_rating = segment_data['rating'].mean()
        avg_price = segment_data['cost_for_two'].mean()
        
        if avg_rating > 4.2 and avg_price > 1500:
            segment_type = "Premium High-Rated"
            color = "#4CAF50"
        elif avg_rating > 4.0 and avg_price < 800:
            segment_type = "Budget Favorites"
            color = "#2196F3"
        elif avg_rating < 3.5:
            segment_type = "Underperformers"
            color = "#FF5252"
        else:
            segment_type = "Average Performers"
            color = "#FFC107"
        
        st.markdown(f"""
        <div class="restaurant-card" style="border-left: 5px solid {color};">
            <h4>Segment {cluster}: {segment_type}</h4>
            <p>📊 {len(segment_data)} restaurants | ⭐ {avg_rating:.2f} avg rating | 💰 ₹{avg_price:.0f} avg price</p>
            <p>📈 Future Score: {segment_data['future_score'].mean():.1f}</p>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown("### 🏆 Top Performers Leaderboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⭐ Top 10 by Current Rating")
        top_rating = df.nlargest(10, 'rating')[['name', 'cuisine', 'rating', 'votes', 'cost_for_two']]
        
        for i, (_, row) in enumerate(top_rating.iterrows(), 1):
            st.markdown(f"""
            <div class="restaurant-card">
                <b>#{i} {row['name']}</b> - {row['cuisine']}<br>
                <span class="badge badge-success">⭐ {row['rating']}</span>
                <span class="badge badge-info">🗳️ {row['votes']} votes</span>
                <span class="badge badge-warning">💰 ₹{row['cost_for_two']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🚀 Top 10 by Future Growth")
        top_growth = df.nlargest(10, 'growth_rate')[['name', 'cuisine', 'growth_rate', 'future_score', 'current_score']]
        
        for i, (_, row) in enumerate(top_growth.iterrows(), 1):
            growth_color = '#4CAF50' if row['growth_rate'] > 20 else '#FFC107'
            st.markdown(f"""
            <div class="restaurant-card">
                <b>#{i} {row['name']}</b> - {row['cuisine']}<br>
                <span class="badge" style="background: {growth_color}; color: white;">📈 {row['growth_rate']}% growth</span>
                <span class="badge badge-info">Current: {row['current_score']}</span>
                <span class="badge badge-success">Future: {row['future_score']}</span>
            </div>
            """, unsafe_allow_html=True)

with tab5:
    st.markdown("### 🎯 Smart Restaurant Recommender")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🔍 Find Your Perfect Restaurant")
        
        cuisine_options = ['All'] + sorted(df['cuisine'].unique().tolist())
        selected_cuisine = st.selectbox("Preferred Cuisine", cuisine_options)
        
        price_range = st.slider("Budget for Two (₹)", 200, 3000, (500, 2000))
        min_rating = st.slider("Minimum Rating", 2.0, 5.0, 4.0, 0.1)
        
        col_a, col_b = st.columns(2)
        with col_a:
            online_req = st.checkbox("Online Order Available")
        with col_b:
            growth_focus = st.checkbox("High Growth Only")
        
        recommend_btn = st.button("🎯 Get Recommendations", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Market Insights")
        
        high_rating_growth = df[df['rating'] > 4]['growth_rate'].mean() - df['growth_rate'].mean()
        online_impact = df[df['online_order'] == 'Yes']['future_score'].mean() - df[df['online_order'] == 'No']['future_score'].mean()
        north_indian_count = len(df[df['cuisine'] == 'North Indian'])
        online_roi = df[df['online_order'] == 'Yes']['growth_rate'].mean() - df[df['online_order'] == 'No']['growth_rate'].mean()
        
        st.markdown(f"""
        <div class="insight-card">
            <h4>💡 Did You Know?</h4>
            <ul>
                <li>Restaurants with <b>rating >4.0</b> have {high_rating_growth:.1f}% higher growth</li>
                <li><b>Online ordering</b> increases future score by {online_impact:.1f} points</li>
                <li><b>North Indian</b> is the most popular cuisine with {north_indian_count} restaurants</li>
                <li>Average ROI for online ordering: <b>{online_roi:.1f}%</b></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if recommend_btn:
        filtered = df.copy()
        
        if selected_cuisine != 'All':
            filtered = filtered[filtered['cuisine'] == selected_cuisine]
        
        filtered = filtered[
            (filtered['cost_for_two'] >= price_range[0]) &
            (filtered['cost_for_two'] <= price_range[1]) &
            (filtered['rating'] >= min_rating)
        ]
        
        if online_req:
            filtered = filtered[filtered['online_order'] == 'Yes']
        
        if growth_focus:
            filtered = filtered[filtered['growth_rate'] > 15]
        
        if len(filtered) > 0:
            filtered = filtered.copy()
            
            max_growth = filtered['growth_rate'].max()
            if max_growth > 0:
                growth_component = filtered['growth_rate'] / max_growth * 30
            else:
                growth_component = 0
            
            filtered['match_score'] = (
                (filtered['rating'] / 5 * 40) +
                growth_component +
                (filtered['online_order'] == 'Yes').astype(int) * 30
            )
            
            filtered = filtered.sort_values('match_score', ascending=False)
            
            st.markdown(f"### Found {len(filtered)} Matching Restaurants")
            
            max_match = filtered['match_score'].max()
            
            for _, row in filtered.head(10).iterrows():
                rating_color = '#4CAF50' if row['rating'] >= 4.2 else '#FFC107' if row['rating'] >= 3.8 else '#FF5252'
                
                st.markdown(f"""
                <div class="restaurant-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div style="flex: 1;">
                            <h4 style="margin: 0 0 5px 0;">{row['name']}</h4>
                            <p style="margin: 0 0 10px 0; color: #666; font-size: 0.95rem;">
                                {row['cuisine']} • {row['location']}
                            </p>
                            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                                <span class="badge badge-info">🗳️ {row['votes']} votes</span>
                                <span class="badge badge-warning">💰 ₹{row['cost_for_two']}</span>
                                <span class="badge badge-success">📈 {row['growth_rate']}% growth</span>
                                <span class="badge {'badge-success' if row['online_order']=='Yes' else 'badge-danger'}">
                                    📱 {row['online_order']}
                                </span>
                            </div>
                        </div>
                        <div style="text-align: right; min-width: 80px;">
                            <div style="font-size: 1.8rem; font-weight: bold; color: {rating_color};">⭐ {row['rating']}</div>
                        </div>
                    </div>
                    <div style="display: flex; gap: 20px; margin-top: 12px; font-size: 0.95rem;">
                        <div>Current Score: <b>{row['current_score']}</b></div>
                        <div>Future Score: <b>{row['future_score']}</b></div>
                        <div>Match: <b>{(row['match_score']/max_match*100):.0f}%</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No restaurants match your criteria. Try adjusting your filters.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
    <p style="font-size: 1.2rem; margin:0;">🚀 Zomato Restaurant Analytics Dashboard v3.0</p>
    <p style="margin:5px 0 0 0;">Advanced Analytics • ML Predictions • Smart Recommendations</p>
</div>
""", unsafe_allow_html=True)