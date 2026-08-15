# FoodRate AI: Restaurant Rating Predictor for the Food-Tech Industry 🍽️

**FoodRate AI** is a complete Python machine learning project and interactive Streamlit web application designed to analyze Zomato restaurant data and accurately predict restaurant **ratings** using **Regression Machine Learning**.

---

## 🌟 Key Features

- **Data Processing & Preprocessing**: Cleaned raw Zomato dataset (51,717 records), parsed ratings formatted as strings (`4.1/5`, `NEW`, `-`), handled missing values, and encoded categorical attributes using `scikit-learn` `ColumnTransformer` pipelines.
- **Data Visualizations**: Interactive Plotly and Matplotlib charts showing rating distributions, cost vs. rating, vote volume impact, top location benchmarks, and feature correlation heatmaps.
- **Machine Learning Regression Pipeline**: Evaluated **Linear Regression** against **Random Forest Regressor** to achieve an **R² Score of 90.5%** and **MAE of 0.064**.
- **Interactive Streamlit Web App**: Multi-page dashboard containing:
  - 🏠 **Home**: Project overview, industry purpose, key performance metrics, and technology stack.
  - 📊 **Dataset Analysis**: Raw data preview, schema datatypes, missing value analysis, and statistical summary.
  - 📈 **Visualizations**: Interactive charts explaining key factors driving restaurant success.
  - 🔮 **Predict Rating**: Dynamic inference form allowing users to select operational parameters and receive instant predicted ratings (e.g. ⭐ **4.25 / 5**).
  - 🎯 **Model Performance**: Detailed evaluation cards (MAE, MSE, RMSE, R² Score), model comparisons, and actual vs. predicted scatter plots.

---

## 📁 Required Project Structure

```text
restaurant-rating-predictor/
│
├── app.py                      # Main Streamlit Web Application
├── train_model.py              # Machine Learning Training & Evaluation Script
├── data/
│   └── enhanced_zomato_dataset_clean.csv  # Zomato Restaurant Dataset
├── models/
│   ├── restaurant_rating_model.pkl   # Serialized Trained Model Pipeline (Joblib Compressed)
│   └── metrics.json                  # Saved Evaluation Metrics & Categorical Metadata
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation
```

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **Pandas** & **NumPy** for data manipulation & preprocessing
- **Scikit-Learn** for Machine Learning pipelines, encoders, and regression models
- **Plotly** & **Matplotlib** for interactive food-tech analytics visualizations
- **Joblib** for model serialization
- **Streamlit** for modern web application UI deployment

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone or download this repository, navigate to the project directory, and install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

Ensure `enhanced_zomato_dataset_clean.csv` is placed inside the `data/` folder:

```text
data/enhanced_zomato_dataset_clean.csv
```

### 3. Train the Machine Learning Model

Run `train_model.py` to preprocess the dataset, train regression models, compute evaluation metrics, and export the trained model pipeline:

```bash
python train_model.py
```

**Expected Console Output:**

```text
============================================================
FoodRate AI: Training Machine Learning Regression Model
============================================================
[+] Loading dataset from 'data/enhanced_zomato_dataset_clean.csv'...
   Initial Dataset Shape: 51717 rows, 17 columns
   Shape after dropping duplicates: 51717 rows
[+] Cleaning target column 'rate'...
   Valid rating rows remaining: 41665 rows
[+] Cleaning numeric feature 'approx_cost'...

[+] Splitting dataset into 80% Training and 20% Testing sets...
   Train set size: 33332 samples
   Test set size:  8333 samples

[+] Training Model 1: Linear Regression...
[Evaluation Results] - Linear Regression:
   * Mean Absolute Error (MAE):     0.2696
   * Mean Squared Error (MSE):      0.1213
   * Root Mean Squared Error (RMSE): 0.3483
   * R2 Score:                       0.3728

[+] Training Model 2: Random Forest Regressor (n_estimators=100)...
[Evaluation Results] - Random Forest Regressor:
   * Mean Absolute Error (MAE):     0.0644
   * Mean Squared Error (MSE):      0.0183
   * Root Mean Squared Error (RMSE): 0.1354
   * R2 Score:                       0.9052

[WINNER] Selected Best Model: Random Forest Regressor (R2 = 0.9052)
[+] Saving final trained model to 'models/restaurant_rating_model.pkl'...
[+] Saved metadata and evaluation metrics to 'models/metrics.json'

[SUCCESS] Training and Model Save Complete!
```

### 4. Run the Streamlit Web Application

Launch the web application locally:

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Model Evaluation Results Summary

| Regressor Model | MAE | MSE | RMSE | R² Score |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | `0.2696` | `0.1213` | `0.3483` | `37.28%` |
| **Random Forest Regressor** | **`0.0644`** | **`0.0183`** | **`0.1354`** | **`90.52%`** |

---

## 🤝 Educational Context

Designed to be accessible, beginner-friendly, and production-ready for **B.Tech CSE / Data Science** projects and portfolio submissions.
