# Machine Learning Capstone Project

A comprehensive machine learning repository implementing end-to-end regression and classification tracks on real-world datasets, complete with cross-validation pipelines, hyperparameter tuning, and metric evaluations.

---

## 📁 Repository Structure

```text
ML_Capstone/
├── data/
│   ├── Loan_approval_data_2025.csv    # Binary classification target (loan_status: 1/0)
│   └── properties_data.csv            # Continuous regression target (property price)
├── notebooks/
│   ├── regression.ipynb               # End-to-end regression track (10 algorithms)
│   └── classification_partA.ipynb     # End-to-end classification track (10 algorithms)
├── apps/                              # Streamlit / Web application deployment files
├── models/                            # Serialized model artifacts (.pkl / .joblib)
├── requirements.txt                   # Environment dependencies
└── README.md                          # Project documentation
```

# 23CSE301 Machine Learning Capstone — Review 1

This repository contains the Jupyter Notebook pipelines and datasets for Review 1 of our Machine Learning Capstone project.
The project is divided into two distinct predictive modeling tracks: a Classification track and a Regression track.

## Track 1: Classification — Loan Status Prediction

- **Objective:** Predict the `loan_status` of an applicant.

- **Dataset:** `Loan_approval_data_2025.csv`.

- **Data Preprocessing & Engineering:**
- Removed duplicate rows and dropped the `customer_id` identifier column.

- Handled outliers in `annual_income`, `loan_amount`, and `current_debt` by capping them at the 1st and 99th percentiles rather than dropping them, to preserve meaningful high-value applicant data.

- Engineered a new feature, `disposable_income_ratio`, to better represent an applicant's absolute repayment capacity.

- Scaled numeric features and one-hot encoded categorical features.

- **Models Evaluated (Part A):**
- Logistic Regression (Baseline).

- K-Nearest Neighbors (KNN).

- Naive Bayes (Gaussian).

- Decision Tree Classifier.

- Support Vector Machine (SVC).

- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-weighted score, and Confusion Matrices.

- **Next Steps:** Ensemble methods (Random Forest, AdaBoost, Gradient Boosting, Bagging) and MLP will be explored in Review 2.

## Track 2: Regression — Real Estate Price Prediction

- **Objective:** Predict real estate property values (target variable: `price`).

- **Dataset:** `properties_data.csv`. The dataset contains 1,905 property records with 38 initial columns.

- **Data Preprocessing & Engineering:**
- Checked for missing values and found zero nulls across the dataset.

- Converted 28 boolean/string amenity features (e.g., `maid_room`, `balcony`, `shared_pool`) into integer 0/1 formats.

- **Data Leakage Mitigation:** Explicitly dropped the `price_per_sqft` column prior to modeling. Because this metric is directly derived from the target variable (`price / size_in_sqft`), including it would artificially inflate the R² score and invalidate the model.

- **Models Evaluated:**
- Linear Regression, Ridge, Lasso, and ElasticNet.

- Polynomial Features.

- Decision Tree Regressor.

- Random Forest Regressor and Gradient Boosting Regressor.

- SVR and KNeighbors Regressor.

- **Evaluation Metrics:** R², Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and 5-fold Cross Validation.

## Data Directory Structure

- `../data/Loan_approval_data_2025.csv`: Primary dataset for the Classification track.

- `../data/properties_data.csv`: Primary dataset for the Regression track.
