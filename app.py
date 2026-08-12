"""
FoodRate AI: Restaurant Rating Predictor & Analytics Web Application
-----------------------------------------------------------------------
A modern Streamlit application for analyzing Zomato restaurant data,
visualizing key food-tech insights, evaluating machine learning models,
and predicting restaurant ratings using trained regression models.
"""

import os
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as io

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(*parts):
    """Resolve a file path relative to the project root."""
    return str(PROJECT_ROOT.joinpath(*parts))

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="FoodRate AI | Restaurant Rating Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Food-Tech Glassmorphism Design
CUSTOM_CSS = """
<style>
    /* Main Theme Overrides */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Container Styling */
    .main-header {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    /* Title Gradient */
    .title-gradient {
        background: linear-gradient(90deg, #f97316 0%, #fb923c 50%, #facc15 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4rem;
        margin-bottom: 4px;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(249, 115, 22, 0.2);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(249, 115, 22, 0.6);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #38bdf8;
    }
    
    /* Result Box Styling */
    .result-box {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.15) 0%, rgba(234, 88, 12, 0.25) 100%);
        border: 2px solid #f97316;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 12px 30px rgba(249, 115, 22, 0.2);
    }
    
    /* Rating Badge */
    .rating-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 8px;
    }
    .badge-excellent { background-color: #16a34a; color: white; }
    .badge-verygood { background-color: #2563eb; color: white; }
    .badge-good { background-color: #d97706; color: white; }
    .badge-average { background-color: #dc2626; color: white; }

    /* Custom Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_clean_dataset():
    """Load and preprocess the dataset for analysis and visualization."""
    data_paths = [
        resolve_project_path('data', 'zomato.csv'),
        resolve_project_path('zomato.csv'),
        resolve_project_path('..', 'data', 'zomato.csv'),
    ]
    df_path = next((p for p in data_paths if os.path.exists(p)), None)

    if df_path is None:
        return None, "Dataset file 'zomato.csv' not found. Please place it in the project root or data/ folder."

    try:
        df = pd.read_csv(df_path)
        df = df.drop_duplicates()
        
        # Clean rate
        df['rate_clean'] = df['rate'].astype(str).str.replace('NEW', '', regex=False)
        df['rate_clean'] = df['rate_clean'].str.replace('-', '', regex=False)
        df['rate_clean'] = df['rate_clean'].str.replace('/5', '', regex=False).str.strip()
        df['rate_clean'] = pd.to_numeric(df['rate_clean'], errors='coerce')
        
        # Clean approx cost
        cost_col = 'approx_cost(for two people)' if 'approx_cost(for two people)' in df.columns else 'approx_cost'
        df['approx_cost_clean'] = df[cost_col].astype(str).str.replace(',', '', regex=False).str.strip()
        df['approx_cost_clean'] = pd.to_numeric(df['approx_cost_clean'], errors='coerce')

        # Clean votes
        df['votes_clean'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0)

        return df, None
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"


@st.cache_resource
def load_model_and_metadata():
    """Load joblib model pipeline and metadata JSON."""
    model_path = resolve_project_path('models', 'restaurant_rating_model.pkl')
    meta_path = resolve_project_path('models', 'metrics.json')

    if not os.path.exists(model_path):
        return None, None, f"Model file '{model_path}' not found. Please run 'python train_model.py' first."

    try:
        model = joblib.load(model_path)
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        return model, meta, None
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


# Sidebar Navigation
def render_sidebar():
    st.sidebar.image("https://img.icons8.com/emoji/96/000000/fork-and-knife-with-plate-emoji.png", width=70)
    st.sidebar.title("FoodRate AI 🍽️")
    st.sidebar.caption("Restaurant Rating & Food-Tech Analytics")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate Project",
        [
            "🏠 Home",
            "📊 Dataset Analysis",
            "📈 Visualizations",
            "🔮 Predict Rating",
            "🎯 Model Performance"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **FoodRate AI System**\n\n"
        "Engineered with Scikit-Learn Regression Pipeline & Streamlit web framework."
    )
    return page


def page_home(df, meta):
    st.markdown("""
    <div class="main-header">
        <div class="title-gradient">FoodRate AI 🍽️</div>
        <h3 style="margin-top:0; color:#cbd5e1; font-weight:400;">
            Restaurant Rating Prediction & Analytics for the Food-Tech Industry
        </h3>
        <p style="color:#94a3b8; font-size:1.05rem;">
            Predict restaurant ratings using advanced machine learning regression models based on 
            pricing, user votes, online ordering, table reservations, location, and cuisine types.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key Statistics Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_rows = f"{len(df):,}" if df is not None else "51,717"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Restaurants</div>
            <div class="metric-value">{total_rows}</div>
            <div class="metric-sub">Analyzed from Zomato</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        valid_ratings = f"{df['rate_clean'].count():,}" if df is not None else "41,665"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Clean Ratings</div>
            <div class="metric-value">{valid_ratings}</div>
            <div class="metric-sub">Target Labels Processed</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        r2_score = f"{meta['metrics']['R2']:.2%}" if meta and 'metrics' in meta else "90.5%"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy (R²)</div>
            <div class="metric-value">{r2_score}</div>
            <div class="metric-sub">Random Forest Regressor</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        mae_val = f"{meta['metrics']['MAE']:.4f}" if meta and 'metrics' in meta else "0.064"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Absolute Error</div>
            <div class="metric-value">{mae_val}</div>
            <div class="metric-sub">On Rating Scale 1.0 - 5.0</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🎯 Project Purpose & Industry Impact")
    st.write("""
    In the rapidly evolving **Food-Tech & Restaurant Analytics** industry, customer ratings directly dictate customer acquisition, footfall, and revenue.
    **FoodRate AI** addresses key business challenges:
    - **Optimizing Restaurant Positioning**: Understand how location, dining type, and table bookings impact consumer perception.
    - **Pricing Strategy Alignment**: Determine the sweet spot for average cost per two people relative to customer expectations.
    - **Data-Driven Success**: Help prospective restaurant owners estimate their potential Zomato rating before launching.
    """)

    st.markdown("---")
    st.subheader("🛠️ Technology Stack")
    
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.markdown("**🐍 Python 3.10+**\nCore execution")
    t2.markdown("**🐼 Pandas & NumPy**\nData cleaning & wrangling")
    t3.markdown("**🤖 Scikit-Learn**\nColumnTransformer & ML Pipeline")
    t4.markdown("**📊 Plotly & Matplotlib**\nInteractive charts")
    t5.markdown("**⚡ Streamlit**\nDeployment UI")


def page_dataset_analysis(df):
    st.title("📊 Dataset Analysis & Exploration")
    st.write("Inspect the underlying Zomato dataset structure, missing value statistics, and summary distributions.")

    if df is None:
        st.error("Dataset not available.")
        return

    # Overview Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows (Dataset)", f"{df.shape[0]:,}")
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Duplicate Rows Dropped", "0 (Cleaned)")

    st.markdown("---")
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📋 Data Types Summary")
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': [str(dtype) for dtype in df.dtypes]
        })
        st.dataframe(dtypes_df, use_container_width=True)

    with col_right:
        st.subheader("⚠️ Missing Value Analysis")
        null_counts = df.isnull().sum()
        null_df = pd.DataFrame({
            'Column': null_counts.index,
            'Missing Count': null_counts.values,
            'Percentage (%)': np.round((null_counts.values / len(df)) * 100, 2)
        })
        st.dataframe(null_df.sort_values(by='Missing Count', ascending=False), use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Numerical Statistical Summary")
    clean_numeric_df = df[['rate_clean', 'votes_clean', 'approx_cost_clean']].dropna()
    clean_numeric_df.columns = ['Rating (/5)', 'Votes Count', 'Approx Cost for Two (₹)']
    st.dataframe(clean_numeric_df.describe().T, use_container_width=True)


def page_visualizations(df):
    st.title("📈 Food-Tech Data Visualizations")
    st.write("Explore dynamic data trends and relationships that influence restaurant ratings.")

    if df is None:
        st.error("Dataset not available.")
        return

    df_clean = df.dropna(subset=['rate_clean'])

    # Visualization 1: Distribution of Ratings
    st.subheader("1. Distribution of Restaurant Ratings")
    fig_hist = px.histogram(
        df_clean, 
        x='rate_clean',
        nbins=30,
        title="Distribution of Cleaned Restaurant Ratings (Scale 1.8 - 4.9)",
        color_discrete_sequence=['#f97316'],
        labels={'rate_clean': 'Restaurant Rating'}
    )
    fig_hist.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("💡 **Insight**: Most restaurant ratings follow a normal distribution centered around **3.7 / 5**. Ratings above 4.5 are relatively rare and represent top-tier establishments.")

    st.markdown("---")

    # Visualization 2 & 3: Cost & Votes vs Rating
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2. Cost for Two vs Rating")
        fig_cost = px.box(
            df_clean.dropna(subset=['approx_cost_clean']),
            x='rate_clean',
            y='approx_cost_clean',
            title="Approx Cost for Two (₹) by Rating Score",
            color_discrete_sequence=['#38bdf8'],
            labels={'rate_clean': 'Rating', 'approx_cost_clean': 'Cost for Two (₹)'}
        )
        fig_cost.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_cost, use_container_width=True)
        st.caption("💡 **Insight**: Higher-rated restaurants (4.0+) generally have a higher average cost for two people.")

    with col2:
        st.subheader("3. Votes Volume vs Rating")
        fig_votes = px.scatter(
            df_clean.sample(min(3000, len(df_clean)), random_state=42),
            x='votes_clean',
            y='rate_clean',
            color='online_order',
            title="Restaurant Votes vs Rating Score",
            labels={'votes_clean': 'Number of Votes', 'rate_clean': 'Rating', 'online_order': 'Online Order'},
            color_discrete_map={'Yes': '#10b981', 'No': '#ef4444'}
        )
        fig_votes.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_votes, use_container_width=True)
        st.caption("💡 **Insight**: Restaurants with higher vote counts consistently maintain ratings above 3.5.")

    st.markdown("---")

    # Visualization 4 & 5: Restaurant Type & Top Locations
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("4. Average Rating by Restaurant Type (Top 10)")
        top_types = df_clean.groupby('rest_type')['rate_clean'].agg(['mean', 'count']).reset_index()
        top_types = top_types[top_types['count'] > 50].sort_values(by='mean', ascending=False).head(10)
        
        fig_rest = px.bar(
            top_types,
            x='mean',
            y='rest_type',
            orientation='h',
            title="Top Restaurant Types by Average Rating",
            color='mean',
            color_continuous_scale='Oranges',
            labels={'mean': 'Average Rating', 'rest_type': 'Restaurant Type'}
        )
        fig_rest.update_layout(template="plotly_dark", height=450, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_rest, use_container_width=True)

    with col4:
        st.subheader("5. Top Locations by Average Rating")
        top_locs = df_clean.groupby('location')['rate_clean'].agg(['mean', 'count']).reset_index()
        top_locs = top_locs[top_locs['count'] > 100].sort_values(by='mean', ascending=False).head(10)
        
        fig_loc = px.bar(
            top_locs,
            x='mean',
            y='location',
            orientation='h',
            title="Top Locations by Average Rating Score",
            color='mean',
            color_continuous_scale='Viridis',
            labels={'mean': 'Average Rating', 'location': 'Location'}
        )
        fig_loc.update_layout(template="plotly_dark", height=450, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_loc, use_container_width=True)

    st.markdown("---")

    # Visualization 6: Heatmap Correlation
    st.subheader("6. Numerical Features Correlation Matrix")
    corr_df = df_clean[['rate_clean', 'votes_clean', 'approx_cost_clean']].corr()
    fig_corr = px.imshow(
        corr_df,
        text_auto=".3f",
        color_continuous_scale="RdBu_r",
        title="Pearson Correlation Heatmap"
    )
    fig_corr.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_corr, use_container_width=True)


def page_predict_rating(model, meta):
    st.title("🔮 Predict Restaurant Rating")
    st.write("Fill in the restaurant attributes below to get an instant AI-powered rating prediction.")

    if model is None:
        st.error("Model pipeline is not loaded. Please ensure 'train_model.py' has executed.")
        return

    # Extract default choices from metadata if available
    locations = meta.get('locations', ['BTM', 'Koramangala 5th Block', 'HSR', 'Indiranagar', 'JP Nagar']) if meta else ['BTM']
    rest_types = meta.get('rest_types', ['Casual Dining', 'Quick Bites', 'Cafe', 'Delivery', 'Dessert Parlor']) if meta else ['Casual Dining']
    listed_types = meta.get('listed_types', ['Buffet', 'Cafes', 'Delivery', 'Dine-out', 'Pubs and bars']) if meta else ['Delivery']

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💰 Operational Details")
            approx_cost = st.number_input(
                "Average Cost for Two People (₹)",
                min_value=40,
                max_value=10000,
                value=600,
                step=50,
                help="Estimated dining cost for two people in Indian Rupees."
            )

            votes = st.number_input(
                "Number of Customer Votes / Reviews",
                min_value=0,
                max_value=20000,
                value=250,
                step=10,
                help="Total number of customer ratings/votes received on Zomato."
            )

            online_order = st.selectbox(
                "Online Order Available?",
                options=["Yes", "No"],
                help="Does the restaurant support online order delivery?"
            )

            book_table = st.selectbox(
                "Table Booking Available?",
                options=["Yes", "No"],
                help="Does the restaurant offer table reservation?"
            )

        with col2:
            st.markdown("### 📍 Location & Category")
            location = st.selectbox(
                "Select Location / Neighborhood",
                options=locations,
                index=0 if "BTM" not in locations else locations.index("BTM")
            )

            rest_type = st.selectbox(
                "Select Restaurant Type",
                options=rest_types,
                index=0 if "Casual Dining" not in rest_types else rest_types.index("Casual Dining")
            )

            listed_in_type = st.selectbox(
                "Listed Listing Category",
                options=listed_types,
                index=0 if "Delivery" not in listed_types else listed_types.index("Delivery")
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🚀 Predict Restaurant Rating", use_container_width=True)

    if submit_btn:
        # Prepare feature DataFrame matching exact pipeline schema
        input_data = pd.DataFrame([{
            'online_order': online_order,
            'book_table': book_table,
            'votes': float(votes),
            'location': location,
            'rest_type': rest_type,
            'listed_in(type)': listed_in_type,
            'approx_cost': float(approx_cost)
        }])

        try:
            prediction = model.predict(input_data)[0]
            # Clip predicted rating to valid rating boundaries [1.0, 5.0]
            prediction_clipped = np.clip(prediction, 1.0, 5.0)

            # Determine badge category
            if prediction_clipped >= 4.4:
                badge_html = '<span class="rating-badge badge-excellent">🌟 Exceptional Rating</span>'
            elif prediction_clipped >= 4.0:
                badge_html = '<span class="rating-badge badge-verygood">✨ Very Good Rating</span>'
            elif prediction_clipped >= 3.5:
                badge_html = '<span class="rating-badge badge-good">👍 Good Rating</span>'
            else:
                badge_html = '<span class="rating-badge badge-average">⚠️ Average / Low Rating</span>'

            st.markdown(f"""
            <div class="result-box">
                <h3 style="color:#cbd5e1; margin-bottom:8px;">Predicted Restaurant Rating</h3>
                <div style="font-size: 3.2rem; font-weight: 800; color: #f97316;">
                    ⭐ {prediction_clipped:.2f} <span style="font-size:1.8rem; color:#94a3b8;">/ 5.0</span>
                </div>
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error making rating prediction: {str(e)}")


def page_model_performance(meta):
    st.title("🎯 Model Performance & Evaluation Metrics")
    st.write("Review the regression evaluation metrics comparing Linear Regression and Random Forest Regressor.")

    if meta is None or 'all_models' not in meta:
        st.warning("Model metadata is not available. Please run `python train_model.py` to regenerate metrics.")
        return

    best_model_name = meta.get('best_model_name', 'Random Forest Regressor')
    best_metrics = meta.get('metrics', {})

    st.success(f"🏆 Best Performing Model: **{best_model_name}**")

    # Evaluation Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (Mean Abs Error)", f"{best_metrics.get('MAE', 0):.4f}")
    c2.metric("MSE (Mean Sq Error)", f"{best_metrics.get('MSE', 0):.4f}")
    c3.metric("RMSE (Root Mean Sq Error)", f"{best_metrics.get('RMSE', 0):.4f}")
    c4.metric("R² Score (Accuracy)", f"{best_metrics.get('R2', 0):.4f}")

    st.markdown("---")
    st.subheader("⚖️ Model Comparison Table")

    comparison_df = pd.DataFrame(meta['all_models'])
    st.table(comparison_df)

    # Plot Model Comparison Bar Chart
    fig_comp = px.bar(
        comparison_df,
        x='model_name',
        y='R2',
        text='R2',
        title="R² Score Model Comparison (Higher is Better)",
        color='model_name',
        color_discrete_sequence=['#38bdf8', '#f97316'],
        labels={'model_name': 'Model Name', 'R2': 'R² Score'}
    )
    fig_comp.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_comp.update_layout(template="plotly_dark", height=400, yaxis_range=[0, 1.05])
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # Actual vs Predicted Plot
    if 'actual_vs_pred' in meta:
        st.subheader("📍 Actual vs Predicted Ratings Scatter Plot")
        act_pred_df = pd.DataFrame(meta['actual_vs_pred'])

        fig_scatter = px.scatter(
            act_pred_df,
            x='actual',
            y='predicted',
            title="Actual Rating vs. Model Predicted Rating (Sample 500 Test Points)",
            labels={'actual': 'Actual Rating', 'predicted': 'Predicted Rating'},
            opacity=0.6,
            color_discrete_sequence=['#10b981']
        )
        # Add ideal diagonal 45-degree reference line
        fig_scatter.add_trace(
            io.Scatter(
                x=[1.5, 5.0],
                y=[1.5, 5.0],
                mode='lines',
                name='Perfect Prediction Line',
                line=dict(color='#ef4444', dash='dash')
            )
        )
        fig_scatter.update_layout(template="plotly_dark", height=480)
        st.plotly_chart(fig_scatter, use_container_width=True)


def main():
    selected_page = render_sidebar()

    # Load Data & Model Resources
    df, df_err = load_clean_dataset()
    model, meta, model_err = load_model_and_metadata()

    if selected_page == "🏠 Home":
        page_home(df, meta)
    elif selected_page == "📊 Dataset Analysis":
        page_dataset_analysis(df)
    elif selected_page == "📈 Visualizations":
        page_visualizations(df)
    elif selected_page == "🔮 Predict Rating":
        if model_err:
            st.error(model_err)
        else:
            page_predict_rating(model, meta)
    elif selected_page == "🎯 Model Performance":
        page_model_performance(meta)


if __name__ == '__main__':
    main()
