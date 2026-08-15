"""
FoodRate AI: Restaurant Rating Predictor - Model Training Script
------------------------------------------------------------------
This script cleans the Zomato restaurant dataset, performs restaurant-level aggregation,
executes feature preprocessing, trains regression machine learning models
(Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor),
evaluates model performance using standard metrics (MAE, MSE, RMSE, R2 Score),
and exports the best-performing pipeline model for Streamlit deployment.
"""

import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent


def find_dataset_path():
    """Locate the Zomato CSV dataset in standard project paths."""
    possible_paths = [
        PROJECT_ROOT / 'data' / 'enhanced_zomato_dataset_clean.csv',
        PROJECT_ROOT / 'enhanced_zomato_dataset_clean.csv',
        PROJECT_ROOT / 'data' / 'zomato.csv',
        PROJECT_ROOT / 'zomato.csv',
        PROJECT_ROOT.parent / 'data' / 'enhanced_zomato_dataset_clean.csv',
        PROJECT_ROOT.parent / 'data' / 'zomato.csv'
    ]
    for path in possible_paths:
        if path.exists():
            return str(path)

    # Fallback search for any .csv file in data/ or PROJECT_ROOT
    for search_dir in [PROJECT_ROOT / 'data', PROJECT_ROOT, PROJECT_ROOT.parent / 'data']:
        if search_dir.exists():
            csv_files = list(search_dir.glob('*.csv'))
            if csv_files:
                return str(csv_files[0])

    raise FileNotFoundError("Could not find dataset ('enhanced_zomato_dataset_clean.csv' or 'zomato.csv') in data/ or project root.")


def load_and_preprocess_data(file_path):
    """
    Load Zomato dataset and clean features across both enhanced and legacy schemas.
    Aggregates menu item rows into distinct restaurant entity records when item-level data is provided.
    """
    print(f"[+] Loading dataset from '{file_path}'...")
    raw_df = pd.read_csv(file_path)
    raw_df.columns = [str(c).strip().replace('\ufeff', '') for c in raw_df.columns]
    print(f"   Initial Raw Dataset Shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    col_map = {c.lower(): c for c in raw_df.columns}

    # Check if dataset is item-level (e.g. enhanced dataset with Item_Name & Restaurant_Name)
    is_item_level = 'item_name' in col_map and 'restaurant_name' in col_map

    if is_item_level:
        print("[+] Processing Item-Level Dataset: Aggregating into unique restaurant records...")
        rest_name_col = col_map['restaurant_name']
        place_col = col_map.get('place_name', rest_name_col)
        city_col = col_map.get('city', place_col)

        # Group by restaurant entity keys
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

        # Target rating
        if 'average_rating' in col_map:
            df['rate'] = grouped[col_map['average_rating']].apply(first_valid)
        elif 'dining_rating' in col_map:
            df['rate'] = grouped[col_map['dining_rating']].apply(first_valid)
        else:
            df['rate'] = np.nan

        # Approx cost for two
        if 'avg_price_restaurant' in col_map:
            avg_p = grouped[col_map['avg_price_restaurant']].apply(first_valid)
            df['approx_cost'] = avg_p * 2.0
        elif 'prices' in col_map:
            df['approx_cost'] = grouped[col_map['prices']].mean() * 2.0
        else:
            df['approx_cost'] = 500.0

        # Votes
        if 'total_votes' in col_map:
            df['votes'] = grouped[col_map['total_votes']].apply(first_valid).fillna(0)
        elif 'votes' in col_map:
            df['votes'] = grouped[col_map['votes']].mean().fillna(0)
        else:
            df['votes'] = 0.0

        # Delivery votes & dining votes for online_order & book_table
        delivery_v = grouped[col_map['delivery_votes']].apply(first_valid).fillna(0) if 'delivery_votes' in col_map else pd.Series(0, index=df.index)
        dining_v = grouped[col_map['dining_votes']].apply(first_valid).fillna(0) if 'dining_votes' in col_map else pd.Series(0, index=df.index)

        df['online_order'] = delivery_v.apply(lambda v: 'Yes' if v > 0 else 'No')
        df['book_table'] = dining_v.apply(lambda v: 'Yes' if v > 0 else 'No')

        df = df.reset_index(drop=True)
    else:
        print("[+] Processing Standard Dataset Schema...")
        df = raw_df.drop_duplicates().copy()

        # Clean target 'rate'
        target_col = None
        for candidate in ['rate', 'average_rating', 'dining_rating', 'rating', 'avg_rating_restaurant', 'delivery_rating']:
            if candidate in col_map:
                target_col = col_map[candidate]
                break

        if target_col is not None:
            df['rate'] = df[target_col].astype(str)
            df['rate'] = df['rate'].str.replace('NEW', '', regex=False)
            df['rate'] = df['rate'].str.replace('-', '', regex=False)
            df['rate'] = df['rate'].str.replace('/5', '', regex=False).str.strip()
            df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
        else:
            df['rate'] = np.nan

        # Clean cost
        cost_col = None
        for candidate in ['approx_cost(for two people)', 'approx_cost', 'avg_price_restaurant', 'prices', 'price']:
            if candidate in col_map:
                cost_col = col_map[candidate]
                break

        if cost_col is not None:
            df['approx_cost'] = df[cost_col].astype(str).str.replace(',', '', regex=False).str.strip()
            df['approx_cost'] = pd.to_numeric(df['approx_cost'], errors='coerce')
        else:
            df['approx_cost'] = 500.0

        # Clean votes
        votes_col = None
        for candidate in ['votes', 'total_votes', 'dining_votes', 'delivery_votes']:
            if candidate in col_map:
                votes_col = col_map[candidate]
                break

        if votes_col is not None:
            df['votes'] = pd.to_numeric(df[votes_col], errors='coerce').fillna(0)
        else:
            df['votes'] = 0.0

        if 'location' not in df.columns:
            df['location'] = df[col_map.get('city', col_map.get('place_name', 'location'))] if any(k in col_map for k in ['city', 'place_name']) else 'Unknown'

        if 'rest_type' not in df.columns:
            df['rest_type'] = df[col_map.get('cuisine', col_map.get('cuisines', 'rest_type'))] if any(k in col_map for k in ['cuisine', 'cuisines']) else 'Other'

        if 'listed_in(type)' not in df.columns:
            df['listed_in(type)'] = df['location']

        if 'online_order' not in df.columns:
            df['online_order'] = 'No'

        if 'book_table' not in df.columns:
            df['book_table'] = 'No'

    # Filter out missing ratings
    df = df.dropna(subset=['rate'])
    df['approx_cost'] = df['approx_cost'].fillna(df['approx_cost'].median() if not df['approx_cost'].empty else 500.0)

    # Standardize string columns
    for col in ['online_order', 'book_table', 'location', 'rest_type', 'listed_in(type)']:
        df[col] = df[col].astype(str).fillna('Unknown').str.strip()

    print(f"   Cleaned Restaurant Entity Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def build_pipeline():
    """Build scikit-learn ColumnTransformer and Pipeline for regression."""
    num_cols = ['votes', 'approx_cost']
    cat_cols = ['online_order', 'book_table', 'location', 'rest_type', 'listed_in(type)']

    num_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    return preprocessor, num_cols, cat_cols


def evaluate_model(y_true, y_pred, model_name):
    """Compute and display regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"\n[Evaluation Results] - {model_name}:")
    print(f"   * Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"   * Mean Squared Error (MSE):      {mse:.4f}")
    print(f"   * Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"   * R2 Score:                       {r2:.4f}")

    return {
        'model_name': model_name,
        'MAE': round(float(mae), 4),
        'MSE': round(float(mse), 4),
        'RMSE': round(float(rmse), 4),
        'R2': round(float(r2), 4)
    }


def main():
    print("=" * 60)
    print("FoodRate AI: Training Machine Learning Regression Model")
    print("=" * 60)

    # 1. Load Data
    data_path = find_dataset_path()
    df = load_and_preprocess_data(data_path)

    # 2. Select Features and Target
    features = ['online_order', 'book_table', 'votes', 'location', 'rest_type', 'listed_in(type)', 'approx_cost']
    X = df[features]
    y = df['rate']

    # 3. Build Preprocessor Pipeline
    preprocessor, num_cols, cat_cols = build_pipeline()

    # 4. Train-Test Split (80% train, 20% test)
    print("\n[+] Splitting dataset into 80% Training and 20% Testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Train set size: {X_train.shape[0]} samples")
    print(f"   Test set size:  {X_test.shape[0]} samples")

    # 5. Model 1: Linear Regression
    print("\n[+] Training Model 1: Linear Regression...")
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_metrics = evaluate_model(y_test, lr_preds, "Linear Regression")

    # 6. Model 2: Random Forest Regressor
    print("\n[+] Training Model 2: Random Forest Regressor...")
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_split=4, random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_preds, "Random Forest Regressor")

    # 7. Model 3: Gradient Boosting Regressor
    print("\n[+] Training Model 3: Gradient Boosting Regressor...")
    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42))
    ])
    gb_pipeline.fit(X_train, y_train)
    gb_preds = gb_pipeline.predict(X_test)
    gb_metrics = evaluate_model(y_test, gb_preds, "Gradient Boosting Regressor")

    # 8. Select Best Model
    models_comparison = [lr_metrics, rf_metrics, gb_metrics]
    best_tuple = max([(rf_pipeline, rf_metrics), (gb_pipeline, gb_metrics), (lr_pipeline, lr_metrics)], key=lambda x: x[1]['R2'])
    best_model, best_metrics = best_tuple
    best_name = best_metrics['model_name']

    print(f"\n[WINNER] Selected Best Model: {best_name} (R2 = {best_metrics['R2']})")

    # 9. Save Trained Model Pipeline & Metadata
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'restaurant_rating_model.pkl')
    print(f"\n[+] Saving final trained model to '{model_path}'...")
    joblib.dump(best_model, model_path, compress=3)

    # Save metrics & categorical values metadata for Streamlit App UI
    unique_locations = sorted([loc for loc in df['location'].unique() if loc and loc != 'nan' and loc != 'Unknown'])
    unique_rest_types = sorted([rt for rt in df['rest_type'].unique() if rt and rt != 'nan' and rt != 'Unknown'])
    unique_listed_types = sorted([lt for lt in df['listed_in(type)'].unique() if lt and lt != 'nan' and lt != 'Unknown'])

    # Sample actual vs predicted for evaluation plot in Streamlit app
    sample_indices = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)
    actual_vs_pred = {
        'actual': y_test.iloc[sample_indices].tolist(),
        'predicted': best_model.predict(X_test.iloc[sample_indices]).tolist()
    }

    metadata = {
        'best_model_name': best_name,
        'metrics': best_metrics,
        'all_models': models_comparison,
        'locations': unique_locations,
        'rest_types': unique_rest_types,
        'listed_types': unique_listed_types,
        'cost_min': float(df['approx_cost'].min()),
        'cost_max': float(df['approx_cost'].max()),
        'cost_median': float(df['approx_cost'].median()),
        'votes_min': int(df['votes'].min()),
        'votes_max': int(df['votes'].max()),
        'votes_median': int(df['votes'].median()),
        'rating_min': float(df['rate'].min()),
        'rating_max': float(df['rate'].max()),
        'actual_vs_pred': actual_vs_pred
    }

    metadata_path = os.path.join('models', 'metrics.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Saved metadata and evaluation metrics to '{metadata_path}'")

    print("\n[SUCCESS] Training and Model Save Complete!")


if __name__ == '__main__':
    main()
