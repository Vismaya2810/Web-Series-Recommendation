import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go


st.markdown("""
    <style>
    .main {
        background-color: #141414;
        color: white;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stSidebar {
        background-color: #221f1f;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 3px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #4169E1;
    }
    /* Uniform sidebar button styling with full text visibility */
    .stSidebar .stButton>button {
        width: 100%;
        min-width: 150px;
        text-align: left;
        box-sizing: border-box;
        white-space: normal;
        height: auto;
        line-height: 1.2em;
        margin-bottom: 5px;
        padding: 10px;
    }
    .stTable {
        background-color: #221f1f;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stTable table {
        width: 100%;
        border-collapse: collapse;
    }
    .stTable th, .stTable td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #333;
        color: white;
    }
    .stTable th {
        background-color: #333;
    }
    .header {
        font-size: 2em;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = "Home 🏠"


# Load dataset
@st.cache_data
def load_data():
    try:
        series = pd.read_csv('data/dataset.csv')
        required_cols = ['Title', 'Year', 'IMDb_Rating', 'Content', 'Genre', 'Director', 'Main_Cast']
        if not all(col in series.columns for col in required_cols):
            raise ValueError("Dataset missing required columns")
        np.random.seed(42)
        n_users = 100
        user_ids = range(1, n_users + 1)
        ratings = []
        for user_id in user_ids:
            for idx, row in series.iterrows():
                if np.random.rand() < 0.5:
                    rating = min(5, max(1, round(row['IMDb_Rating'] / 2 + np.random.normal(0, 0.5))))
                    ratings.append({'user_id': user_id, 'series_id': idx, 'rating': rating})
        ratings_df = pd.DataFrame(ratings)
        return series, ratings_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


# Preprocess for CBF
def preprocess_cbf(series):
    series['text_features'] = (
            series['Genre'].fillna('').str.replace(',', ' ') + ' ' +
            series['Content'].fillna('') + ' ' +
            series['Director'].fillna('') + ' ' +
            series['Main_Cast'].fillna('')
    )
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(series['text_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim, series


# Preprocess for CF
def preprocess_cf(ratings, series):
    user_item_matrix = ratings.pivot(index='user_id', columns='series_id', values='rating').fillna(0)
    svd = TruncatedSVD(n_components=20, random_state=42)
    matrix_reduced = svd.fit_transform(csr_matrix(user_item_matrix))
    predicted_ratings = svd.inverse_transform(matrix_reduced)
    predicted_ratings_df = pd.DataFrame(predicted_ratings,
                                        index=user_item_matrix.index,
                                        columns=user_item_matrix.columns)
    return predicted_ratings_df


# Evaluation metrics
def evaluate_recommendations(recommendations, test_ratings, series, predicted_ratings_df, model_type, k=5):
    precision = 0
    n_users = 0
    for user_id in test_ratings['user_id'].unique():
        user_test = test_ratings[test_ratings['user_id'] == user_id]
        user_rec = recommendations.get(user_id, [])
        if not user_rec:
            continue
        top_k = user_rec[:k]
        relevant = sum(1 for item in top_k if item in user_test[user_test['rating'] >= 4]['series_id'].values)
        precision += relevant / k if top_k else 0
        n_users += 1
    precision = precision / n_users if n_users else 0

    all_recommended = set()
    for user_rec in recommendations.values():
        all_recommended.update(user_rec)
    coverage = len(all_recommended) / len(series) if len(series) else 0

    recommended_ratings = [series.loc[int(idx), 'IMDb_Rating'] for idx in all_recommended if idx in series.index]
    novelty = 1 / (np.mean(recommended_ratings) + 1e-10) if recommended_ratings else 0

    rmse = 0
    mae = 0
    if model_type in ['cf', 'hybrid']:
        actual_ratings = []
        predicted_ratings = []
        for _, row in test_ratings.iterrows():
            user_id = row['user_id']
            series_id = row['series_id']
            if user_id in predicted_ratings_df.index and series_id in predicted_ratings_df.columns:
                actual_ratings.append(row['rating'])
                predicted_ratings.append(predicted_ratings_df.loc[user_id, series_id])
        if actual_ratings:
            rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
            mae = mean_absolute_error(actual_ratings, predicted_ratings)

    return precision, coverage, novelty, rmse, mae


# Get CBF recommendations
def get_cbf_recommendations(series_id, cosine_sim, series, top_n=5):
    idx = series_id
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    series_indices = [i[0] for i in sim_scores]
    return series.iloc[series_indices][['Title', 'Genre', 'IMDb_Rating']]


# Get CF recommendations
def get_cf_recommendations(user_id, predicted_ratings_df, series, top_n=5):
    if user_id not in predicted_ratings_df.index:
        return pd.DataFrame(columns=['Title', 'Genre', 'IMDb_Rating'])
    user_ratings = predicted_ratings_df.loc[user_id]
    top_series = user_ratings.sort_values(ascending=False).head(top_n)
    result = series[series.index.isin(top_series.index)][['Title', 'Genre', 'IMDb_Rating']].copy()
    return result


# Hybrid recommendations
def get_hybrid_recommendations(user_id, predicted_ratings_df, cosine_sim, series, top_n=5, alpha=0.5):
    if user_id not in predicted_ratings_df.index:
        return pd.DataFrame(columns=['Title', 'Genre', 'IMDb_Rating'])
    user_ratings = predicted_ratings_df.loc[user_id]
    rated_series = user_ratings[user_ratings > 0].index
    if len(rated_series) == 0:
        return pd.DataFrame(columns=['Title', 'Genre', 'IMDb_Rating'])

    cbf_scores = np.zeros(len(series))
    for series_id in rated_series:
        if series_id in series.index:
            cbf_scores += cosine_sim[series_id] * user_ratings[series_id]
    cbf_scores = cbf_scores / (np.max(cbf_scores) + 1e-10)

    cf_scores = predicted_ratings_df.loc[user_id].values
    hybrid_scores = alpha * cbf_scores + (1 - alpha) * cf_scores

    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]
    result = series.iloc[top_indices][['Title', 'Genre', 'IMDb_Rating']].copy()
    return result


# Load data
with st.spinner("Loading data..."):
    series, ratings = load_data()
    if series is None or ratings is None:
        st.stop()
    cosine_sim, series = preprocess_cbf(series)
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)
    predicted_ratings_df = preprocess_cf(train_ratings, series)

# Sidebar navigation
st.sidebar.title("Navigation 📺")
pages = [
    "Home 🏠",
    "Recommendations Hub 🎬",
    "Hybrid Recommendations 🤝",
    "Content-Based Recommendations 📝",
    "Collaborative Filtering Recommendations 👥",
    "Model Comparison 📊"
]
for idx, p in enumerate(pages):
    if st.sidebar.button(p, key=f"nav_{idx}"):
        st.session_state['page'] = p

# Render page based on session state
page = st.session_state['page']

# Home Page
if page == "Home 🏠":
    st.markdown('<div class="header">Welcome to Malayalam Series Recommender 🎥</div>', unsafe_allow_html=True)
    st.markdown("""
        Discover your next favorite Malayalam series with our personalized recommendations! 
        Powered by advanced algorithms inspired by cutting-edge research, our system tailors suggestions to your taste. 
        Click below to explore recommendations.
    """)
    if st.button("Explore Recommendations 🎬", key="explore_button"):
        st.session_state['page'] = "Recommendations Hub 🎬"

# Recommendations Hub Page
elif page == "Recommendations Hub 🎬":
    st.markdown('<div class="header">Recommendations Hub 🎬</div>', unsafe_allow_html=True)
    st.markdown("Choose a recommendation type to explore personalized Malayalam series suggestions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Hybrid Recommendations 🤝", key="hybrid_button"):
            st.session_state['page'] = "Hybrid Recommendations 🤝"
    with col2:
        if st.button("Content-Based Recommendations 📝", key="cbf_button"):
            st.session_state['page'] = "Content-Based Recommendations 📝"
    with col3:
        if st.button("Collaborative Filtering Recommendations 👥", key="cf_button"):
            st.session_state['page'] = "Collaborative Filtering Recommendations 👥"

# Hybrid Recommendations Page
elif page == "Hybrid Recommendations 🤝":
    st.markdown('<div class="header">Hybrid Recommendations 🤝</div>', unsafe_allow_html=True)
    st.markdown("Get the best of both worlds with combined content-based and collaborative filtering suggestions.")

    user_id = st.number_input("Enter User ID (1-100)", min_value=1, max_value=100, value=1, key="hybrid_user_id")

    if st.button("Get Hybrid Recommendations", key="get_hybrid_button"):
        hybrid_rec = get_hybrid_recommendations(user_id, predicted_ratings_df, cosine_sim, series, top_n=5)
        if not hybrid_rec.empty:
            st.markdown('<div class="stTable">', unsafe_allow_html=True)
            st.table(hybrid_rec[['Title', 'Genre', 'IMDb_Rating']].style.format({'IMDb_Rating': '{:.1f}'}))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("No hybrid recommendations available. 😕")

# Content-Based Recommendations Page
elif page == "Content-Based Recommendations 📝":
    st.markdown('<div class="header">Content-Based Recommendations 📝</div>', unsafe_allow_html=True)
    st.markdown("Find series similar to your favorites based on genre, content, and cast.")

    series_title = st.selectbox("Select a Series", options=series['Title'].tolist(), key="cbf_series")

    if st.button("Get Content-Based Recommendations", key="get_cbf_button"):
        series_id = series[series['Title'] == series_title].index[0]
        cbf_rec = get_cbf_recommendations(series_id, cosine_sim, series, top_n=5)
        if not cbf_rec.empty:
            st.markdown('<div class="stTable">', unsafe_allow_html=True)
            st.table(cbf_rec[['Title', 'Genre', 'IMDb_Rating']].style.format({'IMDb_Rating': '{:.1f}'}))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("No content-based recommendations available. 😕")

# Collaborative Filtering Recommendations Page
elif page == "Collaborative Filtering Recommendations 👥":
    st.markdown('<div class="header">Collaborative Filtering Recommendations 👥</div>', unsafe_allow_html=True)
    st.markdown("Discover series loved by users with similar tastes.")

    user_id = st.number_input("Enter User ID (1-100)", min_value=1, max_value=100, value=1, key="cf_user_id")

    if st.button("Get Collaborative Filtering Recommendations", key="get_cf_button"):
        cf_rec = get_cf_recommendations(user_id, predicted_ratings_df, series, top_n=5)
        if not cf_rec.empty:
            st.markdown('<div class="stTable">', unsafe_allow_html=True)
            st.table(cf_rec[['Title', 'Genre', 'IMDb_Rating']].style.format({'IMDb_Rating': '{:.1f}'}))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("No collaborative filtering recommendations available. 😕")

# Model Comparison Page
elif page == "Model Comparison 📊":
    st.markdown('<div class="header">Model Comparison 📊</div>', unsafe_allow_html=True)
    st.markdown("Evaluate the performance of our recommendation models with key metrics.")


    def get_all_recommendations(predicted_ratings_df, cosine_sim, series, model_type, top_n=5):
        recommendations = {}
        for user_id in predicted_ratings_df.index:
            if model_type == 'cbf':
                rated = train_ratings[train_ratings['user_id'] == user_id]['series_id'].values
                if rated.any():
                    recs = get_cbf_recommendations(rated[0], cosine_sim, series, top_n).index.tolist()
                    recommendations[user_id] = recs
            elif model_type == 'cf':
                recs = get_cf_recommendations(user_id, predicted_ratings_df, series, top_n).index.tolist()
                recommendations[user_id] = recs
            elif model_type == 'hybrid':
                recs = get_hybrid_recommendations(user_id, predicted_ratings_df, cosine_sim, series,
                                                  top_n).index.tolist()
                recommendations[user_id] = recs
        return recommendations


    cbf_recs = get_all_recommendations(predicted_ratings_df, cosine_sim, series, 'cbf')
    cf_recs = get_all_recommendations(predicted_ratings_df, cosine_sim, series, 'cf')
    hybrid_recs = get_all_recommendations(predicted_ratings_df, cosine_sim, series, 'hybrid')

    cbf_metrics = evaluate_recommendations(cbf_recs, test_ratings, series, predicted_ratings_df, 'cbf')
    cf_metrics = evaluate_recommendations(cf_recs, test_ratings, series, predicted_ratings_df, 'cf')
    hybrid_metrics = evaluate_recommendations(hybrid_recs, test_ratings, series, predicted_ratings_df, 'hybrid')

    metrics_df = pd.DataFrame({
        'Model': ['Content-Based', 'Collaborative Filtering', 'Hybrid'],
        'Precision@5': [cbf_metrics[0], cf_metrics[0], hybrid_metrics[0]],
        'Coverage': [cbf_metrics[1], cf_metrics[1], hybrid_metrics[1]],
        'Novelty': [cbf_metrics[2], cf_metrics[2], hybrid_metrics[2]],
        'RMSE': [cbf_metrics[3], cf_metrics[3], hybrid_metrics[3]],
        'MAE': [cbf_metrics[4], cf_metrics[4], hybrid_metrics[4]]
    })

    st.subheader("Evaluation Metrics Table 📈")
    st.table(metrics_df.style.format({
        'Precision@5': '{:.3f}',
        'Coverage': '{:.3f}',
        'Novelty': '{:.3f}',
        'RMSE': '{:.3f}',
        'MAE': '{:.3f}'
    }))

    st.subheader("Metrics Comparison Plot 📉")
    fig_bar = px.bar(
        metrics_df.melt(id_vars='Model', value_vars=['Precision@5', 'Coverage', 'Novelty']),
        x='Model', y='value', color='variable', barmode='group',
        title='Comparison of Precision@5, Coverage, and Novelty',
        labels={'value': 'Metric Value', 'variable': 'Metric'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=metrics_df['Model'], y=metrics_df['RMSE'], name='RMSE', mode='lines+markers'))
    fig_line.add_trace(go.Scatter(x=metrics_df['Model'], y=metrics_df['MAE'], name='MAE', mode='lines+markers'))
    fig_line.update_layout(title='Comparison of RMSE and MAE', xaxis_title='Model', yaxis_title='Error Value')
    st.plotly_chart(fig_line, use_container_width=True)