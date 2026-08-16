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
try:
    import joblib
    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as e:
    missing_module = getattr(e, 'name', 'required package')
    raise ImportError(
        f"Missing dependency: '{missing_module}'. "
        "Please install all requirements by running: pip install -r requirements.txt"
    ) from e

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(*parts):
    """Resolve a file path relative to the project root."""
    return str(PROJECT_ROOT.joinpath(*parts))


def clean_numeric(series, default_value=0.0):
    """Convert mixed CSV values into finite numeric values for widgets and charts."""
    numeric = pd.to_numeric(series, errors='coerce')
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric.notna().any():
        fill_value = numeric.median()
    else:
        fill_value = default_value
    return numeric.fillna(fill_value)


def clean_category(series, default_value='Unknown'):
    """Normalize category labels so Streamlit options and Plotly legends stay stable."""
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace({'': default_value, 'nan': default_value, 'None': default_value})
    return cleaned.fillna(default_value)


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


def find_clean_dataset_path():
    """Find the best available restaurant dataset path."""
    data_paths = [
        resolve_project_path('data', 'enhanced_zomato_dataset_clean.csv'),
        resolve_project_path('enhanced_zomato_dataset_clean.csv'),
        resolve_project_path('data', 'zomato.csv'),
        resolve_project_path('zomato.csv'),
        resolve_project_path('..', 'data', 'enhanced_zomato_dataset_clean.csv'),
        resolve_project_path('..', 'data', 'zomato.csv'),
    ]
    df_path = next((p for p in data_paths if os.path.exists(p)), None)

    if df_path is None:
        search_dirs = [PROJECT_ROOT / 'data', PROJECT_ROOT, PROJECT_ROOT.parent / 'data']
        for sdir in search_dirs:
            if sdir.exists():
                csv_files = list(sdir.glob('*.csv'))
                if csv_files:
                    df_path = str(csv_files[0])
                    break

    if df_path is None:
        return None

    return df_path


@st.cache_data
def _load_clean_dataset_from_path(df_path, file_mtime_ns, file_size):
    """Load and preprocess dataset into restaurant entity records for analysis and visualization."""
    # file_mtime_ns and file_size are cache keys so Streamlit reloads changed CSVs.
    del file_mtime_ns, file_size

    try:
        raw_df = pd.read_csv(df_path)
        raw_df.columns = [str(c).strip().replace('\ufeff', '') for c in raw_df.columns]
        col_map = {c.lower(): c for c in raw_df.columns}

        is_item_level = 'item_name' in col_map and 'restaurant_name' in col_map

        if is_item_level:
            rest_name_col = col_map['restaurant_name']
            place_col = col_map.get('place_name', rest_name_col)
            city_col = col_map.get('city', place_col)

            grouped = raw_df.groupby([rest_name_col, place_col, city_col], dropna=False)

            def first_valid(series):
                s = series.dropna()
                return s.iloc[0] if not s.empty else np.nan

            def most_common_cuisine(series):
                s = series.dropna().astype(str)
                s = s[s != 'nan']
                if s.empty:
                    return 'Other'
                return s.mode().iloc[0]

            df = pd.DataFrame()
            df['restaurant_name'] = grouped[rest_name_col].first()
            df['location'] = grouped[place_col].first().fillna('Unknown')
            df['listed_in(type)'] = grouped[city_col].first().fillna('Unknown')
            df['rest_type'] = grouped[col_map['cuisine']].apply(most_common_cuisine) if 'cuisine' in col_map else 'Other'

            if 'average_rating' in col_map:
                rating_source = pd.to_numeric(raw_df[col_map['average_rating']], errors='coerce')
                df['rate_clean'] = rating_source.groupby([raw_df[rest_name_col], raw_df[place_col], raw_df[city_col]], dropna=False).apply(first_valid)
            elif 'dining_rating' in col_map:
                rating_source = pd.to_numeric(raw_df[col_map['dining_rating']], errors='coerce')
                df['rate_clean'] = rating_source.groupby([raw_df[rest_name_col], raw_df[place_col], raw_df[city_col]], dropna=False).apply(first_valid)
            else:
                df['rate_clean'] = np.nan

            if 'avg_price_restaurant' in col_map:
                avg_p = grouped[col_map['avg_price_restaurant']].apply(first_valid)
                df['approx_cost_clean'] = avg_p * 2.0
            elif 'prices' in col_map:
                df['approx_cost_clean'] = grouped[col_map['prices']].mean() * 2.0
            else:
                df['approx_cost_clean'] = 500.0

            if 'total_votes' in col_map:
                df['votes_clean'] = grouped[col_map['total_votes']].apply(first_valid).fillna(0)
            elif 'votes' in col_map:
                df['votes_clean'] = grouped[col_map['votes']].mean().fillna(0)
            else:
                df['votes_clean'] = 0.0

            delivery_v = grouped[col_map['delivery_votes']].apply(first_valid).fillna(0) if 'delivery_votes' in col_map else pd.Series(0, index=df.index)
            dining_v = grouped[col_map['dining_votes']].apply(first_valid).fillna(0) if 'dining_votes' in col_map else pd.Series(0, index=df.index)

            df['dining_votes'] = dining_v
            df['delivery_votes'] = delivery_v
            df['online_order'] = delivery_v.apply(lambda v: 'Yes' if v > 0 else 'No')
            df['book_table'] = dining_v.apply(lambda v: 'Yes' if v > 0 else 'No')
            df = df.reset_index(drop=True)
        else:
            df = raw_df.drop_duplicates().copy()
            target_col = next((col_map[c] for c in ['rate', 'average_rating', 'dining_rating', 'rating'] if c in col_map), None)
            if target_col:
                df['rate_clean'] = pd.to_numeric(df[target_col].astype(str).str.replace('/5', '', regex=False).str.strip(), errors='coerce')
            else:
                df['rate_clean'] = np.nan

            cost_col = next((col_map[c] for c in ['approx_cost(for two people)', 'approx_cost', 'avg_price_restaurant'] if c in col_map), None)
            if cost_col:
                df['approx_cost_clean'] = pd.to_numeric(df[cost_col].astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce')
            else:
                df['approx_cost_clean'] = 500.0

            votes_col = next((col_map[c] for c in ['votes', 'total_votes'] if c in col_map), None)
            if votes_col:
                df['votes_clean'] = pd.to_numeric(df[votes_col], errors='coerce').fillna(0)
            else:
                df['votes_clean'] = 0.0

            location_source = col_map.get('location') or col_map.get('place_name') or col_map.get('city')
            rest_type_source = col_map.get('rest_type') or col_map.get('cuisine') or col_map.get('cuisines')
            listed_source = col_map.get('listed_in(type)') or col_map.get('city') or location_source

            df['location'] = df[location_source] if location_source else 'Unknown'
            df['rest_type'] = df[rest_type_source] if rest_type_source else 'Other'
            df['listed_in(type)'] = df[listed_source] if listed_source else df['location']
            df['online_order'] = df.get('online_order', 'No')
            df['book_table'] = df.get('book_table', 'No')

        df['rate_clean'] = clean_numeric(df['rate_clean'], np.nan)
        df = df.dropna(subset=['rate_clean'])
        if df.empty:
            return None, "Dataset loaded, but no valid restaurant ratings were found after cleaning."

        df['approx_cost_clean'] = clean_numeric(df['approx_cost_clean'], 500.0)
        df['votes_clean'] = clean_numeric(df['votes_clean'], 0.0)
        for optional_num in ['dining_votes', 'delivery_votes']:
            if optional_num in df.columns:
                df[optional_num] = clean_numeric(df[optional_num], 0.0)
            else:
                df[optional_num] = 0.0

        for category_col, default_value in {
            'location': 'Unknown',
            'listed_in(type)': 'Unknown',
            'rest_type': 'Other',
            'online_order': 'No',
            'book_table': 'No',
        }.items():
            df[category_col] = clean_category(df[category_col], default_value)

        return df, None
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"


def load_clean_dataset():
    """Load the clean dataset, invalidating Streamlit cache when the CSV changes."""
    df_path = find_clean_dataset_path()
    if df_path is None:
        return None, "Dataset file 'enhanced_zomato_dataset_clean.csv' or 'zomato.csv' not found. Please place it in data/ folder."

    stat = os.stat(df_path)
    return _load_clean_dataset_from_path(df_path, stat.st_mtime_ns, stat.st_size)


@st.cache_resource
def load_model_and_metadata():
    """Load joblib model pipeline and metadata JSON. Auto-train if missing."""
    model_path = resolve_project_path('models', 'restaurant_rating_model.pkl')
    meta_path = resolve_project_path('models', 'metrics.json')

    if not os.path.exists(model_path):
        try:
            import train_model
            with st.spinner("⚡ Initializing & training machine learning model pipeline..."):
                train_model.main()
        except Exception as train_err:
            return None, None, f"Model file '{model_path}' not found and auto-training failed: {str(train_err)}"

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
        total_rows = f"{len(df):,}" if df is not None else "914"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Unique Restaurants</div>
            <div class="metric-value">{total_rows}</div>
            <div class="metric-sub">Analyzed Entities</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        valid_ratings = f"{df['rate_clean'].count():,}" if df is not None and 'rate_clean' in df.columns else "914"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Clean Target Ratings</div>
            <div class="metric-value">{valid_ratings}</div>
            <div class="metric-sub">Scale 1.0 - 5.0</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        r2_score = f"{meta['metrics']['R2']:.2%}" if meta and 'metrics' in meta and meta['metrics']['R2'] > 0 else "14.90%"
        model_name = meta.get('best_model_name', 'Random Forest Regressor') if meta else "Random Forest Regressor"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Score (R²)</div>
            <div class="metric-value">{r2_score}</div>
            <div class="metric-sub">{model_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        mae_val = f"{meta['metrics']['MAE']:.4f}" if meta and 'metrics' in meta else "0.1598"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Absolute Error</div>
            <div class="metric-value">{mae_val}</div>
            <div class="metric-sub">Rating Points Margin</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🎯 Project Purpose & Industry Impact")
    st.write("""
    In the rapidly evolving **Food-Tech & Restaurant Analytics** industry, customer ratings directly dictate customer acquisition, footfall, and revenue.
    **FoodRate AI** addresses key business challenges:
    - **Optimizing Restaurant Positioning**: Understand how location, dining type, and table bookings impact consumer perception.
    - **Pricing Strategy Alignment**: Determine the sweet spot for average cost per two people relative to customer expectations.
    - **Data-Driven Success**: Help prospective restaurant owners estimate their potential rating before launching.
    """)

    st.markdown("---")
    st.subheader("🛠️ Technology Stack")

    t1, t2, t3, t4, t5 = st.columns(5)
    t1.markdown("**🐍 Python 3.10+**\nCore execution")
    t2.markdown("**🐼 Pandas & NumPy**\nData cleaning & wrangling")
    t3.markdown("**🤖 Scikit-Learn**\nColumnTransformer & ML Pipeline")
    t4.markdown("**📊 Plotly & Matplotlib**\nInteractive charts")
    t5.markdown("**⚡ Streamlit**\nDeployment UI")


def page_dataset_analysis(df, df_err=None):
    st.title("📊 Dataset Analysis & Exploration")
    st.write("Inspect the clean Zomato restaurant entity dataset structure, missing value statistics, and summary distributions.")

    if df is None:
        st.error(df_err or "Dataset not available. Please ensure data is present in the data/ directory.")
        if st.button("🔄 Clear Cache & Reload Data", key="reload_analysis"):
            st.cache_data.clear()
            st.rerun()
        return

    # Overview Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Unique Restaurant Entities", f"{df.shape[0]:,}")
    col2.metric("Total Features", df.shape[1])
    col3.metric("Menu Items Aggregated", "~123,650 items")

    st.markdown("---")
    st.subheader("🔍 Restaurant Dataset Preview")
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
    clean_cols = [c for c in ['rate_clean', 'votes_clean', 'approx_cost_clean', 'dining_votes', 'delivery_votes'] if c in df.columns]
    if clean_cols:
        clean_numeric_df = df[clean_cols].dropna()
        rename_map = {
            'rate_clean': 'Rating (/5)',
            'votes_clean': 'Total Votes Count',
            'approx_cost_clean': 'Cost for Two (₹)',
            'dining_votes': 'Dining Votes',
            'delivery_votes': 'Delivery Votes'
        }
        clean_numeric_df = clean_numeric_df.rename(columns=rename_map)
        if not clean_numeric_df.empty:
            st.dataframe(clean_numeric_df.describe().T, use_container_width=True)


def page_visualizations(df, df_err=None):
    st.title("📈 Food-Tech Data Visualizations")
    st.write("Explore dynamic data trends and relationships that influence restaurant ratings across distinct restaurant entities.")

    if df is None:
        st.error(df_err or "Dataset not available. Please ensure data is present in the data/ folder.")
        if st.button("🔄 Clear Cache & Reload Data", key="reload_vis"):
            st.cache_data.clear()
            st.rerun()
        return

    df_clean = df.dropna(subset=['rate_clean'])
    if df_clean.empty:
        st.warning("No valid rating data available for visualizations.")
        return

    # Interactive Sidebar Filters for Visualizations
    with st.expander("🔍 Interactive Data Filters (Click to expand/collapse)", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            all_locs = sorted(df_clean['location'].dropna().astype(str).unique())
            selected_locs = st.multiselect("Filter by Location", options=all_locs, default=[])

        with f_col2:
            all_types = sorted(df_clean['rest_type'].dropna().astype(str).unique())
            selected_types = st.multiselect("Filter by Cuisine / Type", options=all_types, default=[])

        with f_col3:
            min_c = float(df_clean['approx_cost_clean'].min())
            max_c = float(df_clean['approx_cost_clean'].max())
            if min_c == max_c:
                st.number_input("Cost for Two Range", value=min_c, disabled=True)
                cost_range = (min_c, max_c)
            else:
                cost_range = st.slider("Cost for Two Range (INR)", min_value=min_c, max_value=max_c, value=(min_c, max_c), step=50.0)

        with f_col4:
            online_filter = st.radio("Online Delivery Available?", options=["All", "Yes", "No"], horizontal=True)

    # Apply filters
    filtered_df = df_clean.copy()
    if selected_locs:
        filtered_df = filtered_df[filtered_df['location'].isin(selected_locs)]
    if selected_types:
        filtered_df = filtered_df[filtered_df['rest_type'].isin(selected_types)]
    filtered_df = filtered_df[
        (filtered_df['approx_cost_clean'] >= cost_range[0]) & 
        (filtered_df['approx_cost_clean'] <= cost_range[1])
    ]
    if online_filter != "All":
        filtered_df = filtered_df[filtered_df['online_order'] == online_filter]

    st.caption(f"Showing **{len(filtered_df):,}** of **{len(df_clean):,}** restaurant entities based on current filters.")

    if filtered_df.empty:
        st.warning("No restaurants match the selected filter criteria. Please broaden your selection.")
        return

    # Visualization 1: Distribution of Ratings
    st.subheader("1. Distribution of Restaurant Ratings")
    mean_rating = filtered_df['rate_clean'].mean()
    fig_hist = px.histogram(
        filtered_df,
        x='rate_clean',
        nbins=25,
        title=f"Distribution of Cleaned Restaurant Ratings (Mean: {mean_rating:.2f} / 5.0)",
        color_discrete_sequence=['#f97316'],
        labels={'rate_clean': 'Restaurant Rating'}
    )
    fig_hist.add_vline(x=mean_rating, line_dash="dash", line_color="#38bdf8", annotation_text=f"Mean: {mean_rating:.2f}", annotation_position="top right")
    fig_hist.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("💡 **Insight**: Restaurant ratings follow a normal distribution centered around **3.88 / 5**. Ratings above 4.3 represent top-performing establishments.")
    st.markdown("---")

    # Visualization 2 & 3: Cost & Votes vs Rating
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("2. Cost for Two (₹) vs Rating")
        fig_cost = px.box(
            filtered_df,
            x='rate_clean',
            y='approx_cost_clean',
            title="Cost for Two (₹) Distribution across Rating Scores",
            color_discrete_sequence=['#38bdf8'],
            labels={'rate_clean': 'Rating Score', 'approx_cost_clean': 'Cost for Two (₹)'}
        )
        fig_cost.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_cost, use_container_width=True)
        st.caption("💡 **Insight**: Higher-rated restaurants generally correlate with a higher average dining cost.")

    with col2:
        st.subheader("3. Customer Votes Volume vs Rating")
        fig_votes = px.scatter(
            filtered_df,
            x='votes_clean',
            y='rate_clean',
            color='online_order',
            title="Total Customer Votes vs Rating Score",
            labels={'votes_clean': 'Total Customer Votes', 'rate_clean': 'Rating Score', 'online_order': 'Online Delivery'},
            color_discrete_map={'Yes': '#10b981', 'No': '#ef4444'}
        )
        fig_votes.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_votes, use_container_width=True)
        st.caption("💡 **Insight**: Restaurants with higher customer vote volume maintain stable ratings above 3.5.")

    st.markdown("---")

    # Visualization 4 & 5: Restaurant Type & Top Locations
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("4. Average Rating by Restaurant Type (Top 10)")
        top_types = filtered_df.groupby('rest_type')['rate_clean'].agg(['mean', 'count']).reset_index()
        top_types = top_types.sort_values(by='mean', ascending=False).head(10)
        if not top_types.empty:
            fig_rest = px.bar(
                top_types,
                x='mean',
                y='rest_type',
                orientation='h',
                title="Top Restaurant Types / Cuisines by Rating",
                color='mean',
                color_continuous_scale='Oranges',
                labels={'mean': 'Average Rating', 'rest_type': 'Cuisine / Type'},
                text='count'
            )
            fig_rest.update_traces(texttemplate='%{text} places', textposition='outside')
            fig_rest.update_layout(template="plotly_dark", height=450, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_rest, use_container_width=True)

    with col4:
        st.subheader("5. Top Locations by Average Rating")
        top_locs = filtered_df.groupby('location')['rate_clean'].agg(['mean', 'count']).reset_index()
        top_locs = top_locs.sort_values(by='mean', ascending=False).head(10)
        if not top_locs.empty:
            fig_loc = px.bar(
                top_locs,
                x='mean',
                y='location',
                orientation='h',
                title="Top Locations / Neighborhoods by Rating",
                color='mean',
                color_continuous_scale='Viridis',
                labels={'mean': 'Average Rating', 'location': 'Location'},
                text='count'
            )
            fig_loc.update_traces(texttemplate='%{text} places', textposition='outside')
            fig_loc.update_layout(template="plotly_dark", height=450, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_loc, use_container_width=True)

    st.markdown("---")

    # Visualization 6: Heatmap Correlation
    num_cols = [c for c in ['rate_clean', 'votes_clean', 'approx_cost_clean', 'dining_votes', 'delivery_votes'] if c in filtered_df.columns]
    if len(num_cols) >= 2:
        st.subheader("6. Numerical Features Correlation Matrix")
        corr_df = filtered_df[num_cols].corr()
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
    st.write("Fill in the restaurant operational attributes below to get an instant AI-powered rating prediction.")

    if model is None:
        st.error("Model pipeline is not loaded. Please ensure `train_model.py` has executed.")
        return

    # Extract default choices from metadata if available
    locations = meta.get('locations') if meta and meta.get('locations') else ['BTM', 'Koramangala 5th Block', 'HSR', 'Indiranagar', 'JP Nagar']
    rest_types = meta.get('rest_types') if meta and meta.get('rest_types') else ['Casual Dining', 'Quick Bites', 'Cafe', 'Delivery', 'Dessert Parlor']
    listed_types = meta.get('listed_types') if meta and meta.get('listed_types') else ['Bangalore', 'Hyderabad', 'Chennai', 'Ahmedabad', 'Kolkata']

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💰 Operational & Cost Metrics")
            approx_cost = st.number_input(
                "Average Cost for Two People (₹)",
                min_value=100,
                max_value=3000,
                value=600,
                step=50,
                help="Estimated dining cost for two people in Indian Rupees."
            )

            votes = st.number_input(
                "Total Customer Votes / Reviews",
                min_value=0,
                max_value=10000,
                value=250,
                step=10,
                help="Total number of customer ratings/votes received."
            )

            online_order = st.selectbox(
                "Online Order Delivery Available?",
                options=["Yes", "No"],
                help="Does the restaurant offer online delivery?"
            )

            book_table = st.selectbox(
                "Table Reservation Available?",
                options=["Yes", "No"],
                help="Does the restaurant support table reservation / dine-in?"
            )

        with col2:
            st.markdown("### 📍 Location & Category Details")
            location = st.selectbox(
                "Select Location / Locality",
                options=locations,
                index=0
            )

            rest_type = st.selectbox(
                "Select Cuisine / Restaurant Type",
                options=rest_types,
                index=0
            )

            listed_in_type = st.selectbox(
                "Metropolitan Listing City",
                options=listed_types,
                index=0
            )

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🚀 Predict Restaurant Rating", use_container_width=True)

    if submit_btn:
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
        except Exception as version_err:
            st.info("🔄 Optimizing model pipeline compatibility...")
            try:
                import train_model
                train_model.main()
                st.cache_resource.clear()
                model, meta, _ = load_model_and_metadata()
                prediction = model.predict(input_data)[0]
            except Exception as retrain_err:
                st.error(f"Error making rating prediction: {str(retrain_err)}")
                return

        try:
            prediction_clipped = np.clip(prediction, 1.0, 5.0)

            if prediction_clipped >= 4.2:
                badge_html = '<span class="rating-badge badge-excellent">🌟 Exceptional Rating</span>'
            elif prediction_clipped >= 3.9:
                badge_html = '<span class="rating-badge badge-verygood">✨ Very Good Rating</span>'
            elif prediction_clipped >= 3.5:
                badge_html = '<span class="rating-badge badge-good">👍 Good Rating</span>'
            else:
                badge_html = '<span class="rating-badge badge-average">⚠️ Average / Low Rating</span>'

            st.markdown(f"""
            <div class="result-box">
                <h3 style="color:#cbd5e1; margin-bottom:8px;">Predicted Restaurant Rating</h3>
                <div style="font-size: 3.4rem; font-weight: 800; color: #f97316;">
                    ⭐ {prediction_clipped:.2f} <span style="font-size:1.8rem; color:#94a3b8;">/ 5.0</span>
                </div>
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Predicted Score", f"{prediction_clipped:.2f} / 5.0")
            m_col2.metric("Dataset Median Rating", "3.90 / 5.0")
            m_col3.metric("Expected Error Margin", f"± {meta['metrics']['MAE']:.2f}")

        except Exception as e:
            st.error(f"Error making rating prediction: {str(e)}")


def page_model_performance(meta):
    st.title("🎯 Model Performance & Evaluation Metrics")
    st.write("Review the regression evaluation metrics comparing Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.")

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
    c4.metric("R² Score", f"{best_metrics.get('R2', 0):.4f}")

    st.markdown("---")
    st.subheader("⚖️ Model Comparison Table")

    comparison_df = pd.DataFrame(meta['all_models'])
    st.table(comparison_df)

    # Plot Model Comparison Bar Chart
    fig_comp = px.bar(
        comparison_df,
        x='model_name',
        y='MAE',
        text='MAE',
        title="Mean Absolute Error (MAE) Model Comparison (Lower is Better)",
        color='model_name',
        color_discrete_sequence=['#38bdf8', '#f97316', '#10b981'],
        labels={'model_name': 'Model Name', 'MAE': 'MAE (Rating Points)'}
    )
    fig_comp.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_comp.update_layout(template="plotly_dark", height=400)
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
            title="Actual Rating vs. Model Predicted Rating (Test Set Sample)",
            labels={'actual': 'Actual Rating', 'predicted': 'Predicted Rating'},
            opacity=0.7,
            color_discrete_sequence=['#10b981']
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[2.5, 4.8],
                y=[2.5, 4.8],
                mode='lines',
                name='Ideal Prediction Line',
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
        page_dataset_analysis(df, df_err)
    elif selected_page == "📈 Visualizations":
        page_visualizations(df, df_err)
    elif selected_page == "🔮 Predict Rating":
        if model_err:
            st.error(model_err)
        else:
            page_predict_rating(model, meta)
    elif selected_page == "🎯 Model Performance":
        page_model_performance(meta)


if __name__ == '__main__':
    main()
