# ML Capstone: Regression, Classification & Clustering

An end-to-end Machine Learning pipeline for training, evaluating, comparing, and deploying models across three core machine learning paradigms: **Regression, Classification, and Clustering**. The project integrates trained models into an interactive **Streamlit dashboard** for live inference, performance comparison, and visual exploration.

## Overview

This project demonstrates an end-to-end machine learning workflow, covering data preprocessing, feature engineering, model training, evaluation, model serialization, and deployment through a web-based interface.

The pipeline supports three major machine learning tasks:

- **Regression:** Predict continuous numerical values using models such as Linear Regression, Random Forest, and XGBoost.
- **Classification:** Predict categorical labels using models such as Logistic Regression, Random Forest, and LightGBM.
- **Clustering:** Discover natural groups and behavioral patterns in unlabeled data using K-Means, Agglomerative Hierarchical Clustering, and DBSCAN.

The trained models are serialized using `.pkl` or `.joblib` files and integrated into a unified Streamlit dashboard, allowing users to explore model performance and generate predictions without directly interacting with the training code.

## Problem Statement

Real-world datasets often require different analytical approaches depending on the objective. Predicting a numerical value, identifying a category, and discovering hidden groups are distinct problems that require different algorithms and evaluation strategies.

This project addresses these requirements through a unified machine learning pipeline.

1. **Regression:** Estimate continuous target values while minimizing prediction errors.
2. **Classification:** Predict class labels while balancing precision, recall, and overall predictive performance.
3. **Clustering:** Identify meaningful groups in data without relying on predefined labels.

The objective is to build a modular and reusable framework that simplifies experimentation, model comparison, and interactive deployment across these tasks.

## Objectives

- Implement three major machine learning paradigms within a single project.
- Perform exploratory data analysis and data preprocessing.
- Handle missing values, categorical variables, and feature scaling.
- Train and compare multiple algorithms for each task.
- Evaluate models using task-specific performance metrics.
- Identify suitable models based on evaluation results.
- Serialize trained models for reuse during inference.
- Build an interactive Streamlit dashboard for predictions and visual exploration.
- Maintain a modular codebase that separates preprocessing, training, and application logic.

## Dataset Description

The pipeline is designed to work with structured datasets containing numerical and categorical features. The datasets can be replaced or configured according to the requirements of each machine learning task.

| Task | Dataset type | Instances | Features | Target |
|---|---|---:|---:|---|
| Regression | Synthetic or housing data | 10,000 | 12 | `target_price` |
| Classification | Customer attrition or loan default | 10,000 | 14 | `churn_status` |
| Clustering | Customer behavioral data | 10,000 | 8 | No predefined target |

### Dataset characteristics

- **Regression:** A continuous target variable is used to predict numerical outcomes.
- **Classification:** A binary target variable represents the class to be predicted.
- **Clustering:** Numerical features are used to identify groups of similar observations without predefined class labels.

> **Note:** The dataset names, sizes, feature counts, and target columns above describe the configured project specification. Replace them with the actual dataset details if your implementation uses different data.

## Machine Learning Workflow

The overall workflow follows these stages:

1. Load the datasets.
2. Inspect the data and perform exploratory data analysis.
3. Clean the data and handle missing values.
4. Encode categorical features and scale numerical features where required.
5. Split labeled datasets into training and testing sets for regression and classification.
6. Train multiple candidate models.
7. Evaluate models using appropriate metrics.
8. Compare model performance and select suitable candidates.
9. Save trained models and preprocessing objects.
10. Load the saved artifacts into the Streamlit application.
11. Generate predictions or explore clustering results through the dashboard.

## Data Preprocessing

Data preprocessing prepares raw data for machine learning algorithms and helps maintain consistency between training and inference.

### 1. Missing Value Imputation

Missing values are handled using:

- **Median imputation:** Used for numerical features, especially when the data contains skewed distributions or outliers.
- **Mode imputation:** Used for categorical features by replacing missing values with the most frequent category.

### 2. Categorical Encoding

Categorical variables are converted into numerical representations.

- **One-Hot Encoding:** Represents low-cardinality categorical features using binary indicator columns.
- **Target Encoding:** Represents categories using statistics derived from the target variable. It must be implemented carefully within training folds to prevent data leakage.

### 3. Feature Scaling

`StandardScaler` standardizes numerical features to approximately zero mean and unit variance.

Scaling is particularly relevant for distance-sensitive or gradient-based algorithms, including:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Support Vector Machines (SVM)
- K-Means Clustering
- Principal Component Analysis (PCA)

Tree-based algorithms generally do not require standardization.

### 4. Train-Test Separation

Regression and classification models are evaluated on data held out from training. Preprocessing steps that learn from the data should be fitted on the training set and then applied to the test set.

Clustering is evaluated using internal clustering metrics and visual analysis because predefined labels are not required.

## Regression

Regression predicts continuous numerical values from input features.

### Algorithms

| Algorithm | Purpose |
|---|---|
| Linear Regression | Establishes a simple baseline model |
| Random Forest Regressor | Captures nonlinear relationships using an ensemble of decision trees |
| XGBoost Regressor | Uses gradient boosting to improve predictive performance |

### Evaluation Metrics

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted values.
- **RMSE (Root Mean Squared Error):** Square root of the average squared prediction error; larger errors receive greater penalties.
- **R² Score:** Measures how well the model explains variation in the target relative to a mean-based baseline.
- **Training Time:** Measures the time required to fit the model.

Lower MAE and RMSE values are generally preferred. Higher R² values indicate a better fit, although performance must also be assessed on unseen data.

### Reported Results

| Model | MAE | RMSE | R² Score | Training Time (s) | Status |
|---|---:|---:|---:|---:|---|
| Linear Regression | 4.12 | 5.84 | 0.721 | 0.08 | Baseline |
| Random Forest Regressor | 2.45 | 3.61 | 0.884 | 4.12 | Candidate |
| **XGBoost Regressor** | **1.98** | **2.91** | **0.923** | **1.45** | **Selected** |

**Reported best model:** XGBoost Regressor, based on the lowest MAE and RMSE and the highest R² score among the compared models.

## Classification

Classification predicts discrete class labels from input features.

### Algorithms

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Provides an interpretable baseline for classification |
| Random Forest Classifier | Combines multiple decision trees to improve predictive performance |
| LightGBM Classifier | Uses gradient-boosted decision trees for efficient classification |

### Evaluation Metrics

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between classes across classification thresholds.

Precision and recall should be considered alongside accuracy, especially when the classes are imbalanced.

### Reported Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.812 | 0.791 | 0.742 | 0.765 | 0.864 | Baseline |
| Random Forest Classifier | 0.887 | 0.865 | 0.841 | 0.853 | 0.932 | Candidate |
| **LightGBM Classifier** | **0.914** | **0.898** | **0.881** | **0.889** | **0.957** | **Selected** |

**Reported best model:** LightGBM Classifier, which achieved the highest accuracy, F1-score, and ROC-AUC among the compared models.

## Clustering

Clustering is an unsupervised learning task that groups observations based on similarities in their feature values.

Unlike regression and classification, clustering does not require a predefined target column.

### Algorithms

| Algorithm | Description |
|---|---|
| K-Means | Partitions observations into a specified number of clusters by minimizing within-cluster squared distances |
| Agglomerative Hierarchical Clustering | Builds clusters by repeatedly merging groups according to a linkage criterion |
| DBSCAN | Identifies dense groups of observations and labels sparse observations as noise |

### Selecting the Number of Clusters

For K-Means, the number of clusters can be explored using:

- **Elbow Method:** Plots within-cluster sum of squares against the number of clusters to identify a potential point of diminishing returns.
- **Silhouette Analysis:** Measures how well observations fit within their assigned clusters compared with other clusters.

The selected number of clusters should be justified using evaluation metrics, visualizations, cluster sizes, and interpretability.

### Clustering Evaluation Metrics

- **Silhouette Score:** Measures cluster cohesion and separation. Higher values are generally better, with a range from -1 to 1.
- **Davies-Bouldin Index:** Measures similarity between clusters. Lower values generally indicate better-separated clusters.
- **Calinski-Harabasz Score:** Compares between-cluster dispersion with within-cluster dispersion. Higher values generally indicate better-defined clusters.

These metrics measure different aspects of cluster quality, so no single metric should determine the final choice.

### Reported Results

| Algorithm | Clusters | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score |
|---|---:|---:|---:|---:|
| **K-Means** | **4** | **0.582** | **0.714** | **4,120.5** |
| Agglomerative Hierarchical | 4 | 0.541 | 0.802 | 3,890.1 |
| DBSCAN | 3 | 0.439 | 0.985 | 2,750.4 |

DBSCAN identifies three clusters in the reported results, with 4.2% of observations labeled as noise.

**Reported best model:** K-Means with four clusters, based on its highest Silhouette Score and Calinski-Harabasz Score and its lowest Davies-Bouldin Index among the listed methods.

### Cluster Interpretation

After clustering, the groups can be profiled by examining their feature distributions and summary statistics.

Possible customer segments might include:

- High-activity customers
- High-spending customers
- Customers who frequently use cash advances
- Low-activity customers

These are illustrative segment labels. Actual labels should be assigned only after examining the cluster profiles and the underlying data.

## Streamlit Dashboard

The project includes a Streamlit application intended to provide an interactive interface for exploring the machine learning pipeline.

### Intended Capabilities

- Select a machine learning task.
- Explore model evaluation results.
- Enter feature values for regression or classification predictions.
- View predicted numerical values or class labels.
- Explore clustering assignments and visualizations.
- Compare candidate models using relevant performance metrics.

The exact functionality depends on the components implemented in `app.py`.

## Repository Structure

```text
.
├── data/
│   ├── raw/                     # Original datasets
│   └── processed/               # Cleaned and transformed datasets
│
├── models/                      # Saved model artifacts
│   ├── regression/
│   ├── classification/
│   └── clustering/
│
├── notebooks/                   # EDA and model experiments
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # Data cleaning and feature engineering
│   ├── train_regression.py      # Regression model training and evaluation
│   ├── train_classification.py  # Classification model training and evaluation
│   └── train_clustering.py      # Clustering and cluster evaluation
│
├── app.py                       # Streamlit application entry point
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

The model subdirectories shown above are a suggested organization. Adjust the structure to match the files in your actual repository.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version
- pip
- Git
- A compatible dataset for each machine learning task

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your actual GitHub repository URL and directory name.

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` has not been created, the project may require packages such as:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn streamlit joblib xgboost lightgbm scipy
```

Install only the dependencies required by your actual implementation.

### 4. Prepare the Datasets

Place the original datasets inside `data/raw/` and configure the training scripts to use the correct file paths.

Make sure that:

- The regression dataset contains the intended continuous target.
- The classification dataset contains the intended class label.
- The clustering dataset contains appropriate numerical or encoded features.
- Identifiers and target columns are handled correctly for each task.

### 5. Train the Models

Run the training scripts from the repository root:

```bash
python src/train_regression.py
python src/train_classification.py
python src/train_clustering.py
```

The scripts should preprocess the relevant data, train candidate models, calculate evaluation metrics, and save the required artifacts to the configured model directory.

The exact artifact filenames depend on the implementation.

### 6. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the dashboard.

> **Important:** The dashboard must be configured to load existing model artifacts. Train the required models first if the application expects serialized files that are not already included.

## Model Serialization

Trained models can be saved using `joblib` or Python's `pickle` module.

Example using `joblib`:

```python
import joblib

# Save a trained model
joblib.dump(model, "models/model.joblib")

# Load the saved model
loaded_model = joblib.load("models/model.joblib")
```

Save the preprocessing pipeline alongside the model when it contains fitted imputers, encoders, scalers, or other transformations. This helps ensure that new input data is processed consistently during inference.

**Security note:** Load serialized model files only from trusted sources. Pickle-based formats can execute code when loaded.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical operations |
| Pandas | Data manipulation and analysis |
| Scikit-learn | Preprocessing, machine learning algorithms, and evaluation |
| XGBoost | Gradient-boosted regression |
| LightGBM | Gradient-boosted classification |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SciPy | Scientific computing and hierarchical clustering |
| Joblib / Pickle | Model serialization |
| Streamlit | Interactive web dashboard |
| Git & GitHub | Version control and project hosting |

## Key Features

- Unified workflow for regression, classification, and clustering.
- Modular scripts for preprocessing and model training.
- Comparison of multiple algorithms for each task.
- Task-specific evaluation metrics.
- Support for numerical and categorical data preprocessing.
- Serialized models for reuse during inference.
- Interactive dashboard for predictions and visual exploration.
- Clustering analysis for discovering potential behavioral segments.
- Extensible structure for experimenting with additional algorithms.

## Limitations

- Performance depends on dataset quality, feature selection, and preprocessing choices.
- Reported results may not generalize to unseen datasets or real-world deployments.
- Target encoding can cause data leakage if it is not performed correctly.
- Clustering results depend on feature scaling, distance measures, hyperparameters, and cluster geometry.
- Internal clustering metrics do not guarantee that discovered segments are meaningful in practice.
- Model training time and prediction latency vary with the hardware, dataset size, and configuration.
- A model artifact must be paired with compatible preprocessing steps and input features.

## Future Improvements

- Add automated hyperparameter tuning and cross-validation.
- Implement experiment tracking and reproducible model comparisons.
- Add feature importance and explainability using SHAP or similar tools.
- Include confidence estimates and prediction diagnostics where appropriate.
- Add robust input validation and clearer error handling to the dashboard.
- Introduce automated data and model validation.
- Experiment with additional regression, classification, and clustering algorithms.
- Add deployment through a cloud hosting platform.
- Add unit tests and continuous integration using GitHub Actions.
- Improve clustering interpretation through interactive cluster profiling.
- Track model versions and dataset versions for reproducibility.
