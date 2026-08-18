# ML Assignment 2 — Wine Quality Classification

## a. Problem Statement

Predict whether a red wine is of **good quality** (quality ≥ 7) or **bad quality** (quality < 7) based on its physicochemical properties. This is a **binary classification** problem where we implement and compare 5 different ML models to identify the best-performing classifier.

## b. Dataset Description

- **Source**: [UCI Machine Learning Repository — Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Type**: Red Wine
- **Instances**: 1,599
- **Features**: 11 physicochemical input features + 1 target

| # | Feature | Description |
|---|---------|-------------|
| 1 | fixed acidity | Tartaric acid concentration (g/dm³) |
| 2 | volatile acidity | Acetic acid concentration (g/dm³) |
| 3 | citric acid | Citric acid concentration (g/dm³) |
| 4 | residual sugar | Remaining sugar after fermentation (g/dm³) |
| 5 | chlorides | Sodium chloride concentration (g/dm³) |
| 6 | free sulfur dioxide | Free SO₂ (mg/dm³) |
| 7 | total sulfur dioxide | Total SO₂ (mg/dm³) |
| 8 | density | Density of wine (g/cm³) |
| 9 | pH | pH level |
| 10 | sulphates | Potassium sulphate concentration (g/dm³) |
| 11 | alcohol | Alcohol content (% vol) |
| Target | quality | Binarized: Good (≥7) = 1, Bad (<7) = 0 |

## c. GitHub Repository Link

> *[Add your GitHub repo link here after pushing]*

## d. Models Used & Evaluation Metrics

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8938 | 0.8804 | 0.6957 | 0.3721 | 0.4848 | 0.4580 |
| Decision Tree | 0.9062 | 0.8182 | 0.6383 | 0.6977 | 0.6667 | 0.6131 |
| KNN | 0.8938 | 0.8237 | 0.6667 | 0.4186 | 0.5143 | 0.4738 |
| Naive Bayes | 0.8594 | 0.8517 | 0.4844 | 0.7209 | 0.5794 | 0.5131 |
| Random Forest (Ensemble) | 0.9375 | 0.9546 | 0.9259 | 0.5814 | 0.7143 | 0.7045 |

### Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Achieves decent accuracy (89.4%) but struggles with recall (37.2%) for the minority "Good" class. The model is conservative in predicting good wines, leading to many false negatives. Suitable as a baseline but limited by the linear decision boundary. |
| Decision Tree | Offers a balanced trade-off between precision (63.8%) and recall (69.8%), resulting in a solid F1 score (0.667). However, it has the lowest AUC (0.818) among all models, indicating weaker probability calibration and potential overfitting to training data. |
| KNN | Performance is similar to Logistic Regression with accuracy of 89.4%. Recall is low (41.9%), meaning it misses many good wines. The distance-based approach is sensitive to feature scaling and the imbalanced class distribution limits its effectiveness. |
| Naive Bayes | Has the highest recall (72.1%) among all models, making it the best at detecting good wines. However, precision is the lowest (48.4%), meaning it produces more false positives. The conditional independence assumption provides a different decision boundary that favors sensitivity over specificity. |
| Random Forest (Ensemble) | **Best overall performer** with the highest accuracy (93.8%), AUC (0.955), precision (92.6%), F1 (0.714), and MCC (0.705). The ensemble of 100 decision trees reduces overfitting and provides robust predictions. Only moderate recall (58.1%) prevents it from catching all good wines, but the very high precision means predictions are highly reliable. |
| **Overall Winner** | **Random Forest** — Dominates across 5 out of 6 metrics (Accuracy, AUC, Precision, F1, MCC). Its ensemble approach effectively handles the class imbalance and captures complex non-linear feature interactions in the wine quality data. |

## How to Run Locally

```bash
# Clone the repository
git clone <your-repo-url>
cd <project-folder>

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train models
python model/train_models.py

# Run Streamlit app
streamlit run app.py
```

## Project Structure

```
project-folder/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── test_data.csv           # Test data for evaluation
└── model/
    ├── train_models.py     # Training script for all 5 models
    ├── scaler.pkl          # Fitted StandardScaler
    ├── feature_names.pkl   # Feature column names
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── results.csv         # Evaluation metrics summary
```

## Streamlit App Features

- **CSV Upload**: Upload test data for evaluation
- **Model Selection**: Dropdown to choose between 5 classification models
- **Metrics Display**: Accuracy, AUC, Precision, Recall, F1, MCC
- **Confusion Matrix**: Visual heatmap of predictions
- **Classification Report**: Detailed per-class metrics
- **Model Comparison**: Side-by-side comparison of all models

## Live Streamlit App

> *https://2025ac05274-ml-assignment2.streamlit.app*
