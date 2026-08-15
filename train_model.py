"""
FoodRate AI: Restaurant Rating Predictor - Model Training Script
------------------------------------------------------------------
This script cleans the Zomato restaurant dataset, performs feature preprocessing,
trains regression machine learning models (Linear Regression & Random Forest Regressor),
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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
    raise FileNotFoundError("Could not find dataset ('enhanced_zomato_dataset_clean.csv' or 'zomato.csv') in data/ or project root.")


def load_and_preprocess_data(file_path):
    """
    Load Zomato dataset and clean features across both enhanced and legacy schemas:
    - Drop duplicate rows
    - Handle rating target ('rate', 'Average_Rating', 'Dining_Rating')
    - Handle cost feature ('approx_cost', 'approx_cost(for two people)', 'Avg_Price_Restaurant', 'Prices')
    - Handle votes feature ('votes', 'Total_Votes', 'Votes')
    - Handle location, rest_type, listed_in(type), online_order, book_table
    """
    print(f"[+] Loading dataset from '{file_path}'...")
    df = pd.read_csv(file_path)
    initial_shape = df.shape
    print(f"   Initial Dataset Shape: {initial_shape[0]} rows, {initial_shape[1]} columns")

    # 1. Remove duplicate rows
    df = df.drop_duplicates()
    print(f"   Shape after dropping duplicates: {df.shape[0]} rows")

    # 2. Clean Target Column: 'rate'
    print("[+] Cleaning target column 'rate'...")
    if 'rate' in df.columns:
        df['rate'] = df['rate'].astype(str)
        df['rate'] = df['rate'].str.replace('NEW', '', regex=False)
        df['rate'] = df['rate'].str.replace('-', '', regex=False)
        df['rate'] = df['rate'].str.replace('/5', '', regex=False)
        df['rate'] = df['rate'].str.strip()
        df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    elif 'Average_Rating' in df.columns:
        df['rate'] = pd.to_numeric(df['Average_Rating'], errors='coerce')
    elif 'Dining_Rating' in df.columns:
        df['rate'] = pd.to_numeric(df['Dining_Rating'], errors='coerce')

    # Drop rows where target rating is NaN
    df = df.dropna(subset=['rate'])
    print(f"   Valid rating rows remaining: {df.shape[0]} rows")

    # 3. Clean Feature: 'approx_cost'
    print("[+] Cleaning numeric feature 'approx_cost'...")
    if 'approx_cost(for two people)' in df.columns:
        cost_col = 'approx_cost(for two people)'
        df['approx_cost'] = df[cost_col].astype(str).str.replace(',', '', regex=False).str.strip()
        df['approx_cost'] = pd.to_numeric(df['approx_cost'], errors='coerce')
    elif 'approx_cost' in df.columns:
        df['approx_cost'] = pd.to_numeric(df['approx_cost'].astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce')
    elif 'Avg_Price_Restaurant' in df.columns:
        df['approx_cost'] = pd.to_numeric(df['Avg_Price_Restaurant'], errors='coerce')
    elif 'Prices' in df.columns:
        df['approx_cost'] = pd.to_numeric(df['Prices'], errors='coerce')

    # 4. Clean votes feature
    if 'votes' in df.columns:
        df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0)
    elif 'Total_Votes' in df.columns:
        df['votes'] = pd.to_numeric(df['Total_Votes'], errors='coerce').fillna(0)
    elif 'Votes' in df.columns:
        df['votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0)

    # 5. Handle missing values & categorical mappings
    if 'location' not in df.columns:
        if 'Place_Name' in df.columns:
            df['location'] = df['Place_Name']
        elif 'City' in df.columns:
            df['location'] = df['City']

    if 'rest_type' not in df.columns:
        if 'Cuisine' in df.columns:
            df['rest_type'] = df['Cuisine']

    if 'listed_in(type)' not in df.columns:
        if 'City' in df.columns:
            df['listed_in(type)'] = df['City']
        elif 'Cuisine' in df.columns:
            df['listed_in(type)'] = df['Cuisine']

    if 'online_order' not in df.columns:
        if 'Is_Bestseller' in df.columns:
            df['online_order'] = df['Is_Bestseller'].map({1: 'Yes', 0: 'No'})
        elif 'Best_Seller' in df.columns:
            df['online_order'] = df['Best_Seller'].apply(lambda x: 'Yes' if str(x).upper() in ['YES', 'BESTSELLER', 'MUST TRY'] else 'No')
        else:
            df['online_order'] = 'No'

    if 'book_table' not in df.columns:
        if 'Is_Highly_Rated' in df.columns:
            df['book_table'] = df['Is_Highly_Rated'].map({1: 'Yes', 0: 'No'})
        else:
            df['book_table'] = 'No'

    for col in ['online_order', 'book_table', 'location', 'rest_type', 'listed_in(type)']:
        df[col] = df[col].astype(str).fillna('Unknown').str.strip()

    return df


def build_pipeline():
    """Build scikit-learn ColumnTransformer and Pipeline for regression."""
    num_cols = ['votes', 'approx_cost']
    cat_cols = ['online_order', 'book_table', 'location', 'rest_type', 'listed_in(type)']

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
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

    # 5. Train Linear Regression Model
    print("\n[+] Training Model 1: Linear Regression...")
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_metrics = evaluate_model(y_test, lr_preds, "Linear Regression")

    # 6. Train Random Forest Regressor Model
    print("\n[+] Training Model 2: Random Forest Regressor (n_estimators=100)...")
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_preds = rf_pipeline.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_preds, "Random Forest Regressor")

    # 7. Select Best Model
    models_comparison = [lr_metrics, rf_metrics]
    best_model = rf_pipeline if rf_metrics['R2'] >= lr_metrics['R2'] else lr_pipeline
    best_name = rf_metrics['model_name'] if rf_metrics['R2'] >= lr_metrics['R2'] else lr_metrics['model_name']
    best_metrics = rf_metrics if rf_metrics['R2'] >= lr_metrics['R2'] else lr_metrics

    print(f"\n[WINNER] Selected Best Model: {best_name} (R2 = {best_metrics['R2']})")

    # 8. Save Trained Model Pipeline & Metadata
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'restaurant_rating_model.pkl')
    print(f"\n[+] Saving final trained model (compressed for GitHub limit compliance) to '{model_path}'...")
    joblib.dump(best_model, model_path, compress=3)

    # Save metrics & categorical values metadata for Streamlit App UI
    unique_locations = sorted([loc for loc in df['location'].dropna().unique() if loc != 'nan' and loc != 'Unknown'])
    unique_rest_types = sorted([rt for rt in df['rest_type'].dropna().unique() if rt != 'nan' and rt != 'Unknown'])
    unique_listed_types = sorted([lt for lt in df['listed_in(type)'].dropna().unique() if lt != 'nan' and lt != 'Unknown'])

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
