"""
ML Assignment 2 - Training Script
Trains 5 classification models on the Wine Quality dataset and saves them.
Outputs evaluation metrics and exports test_data.csv.
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)


def load_and_preprocess_data():
    """Load Wine Quality dataset and preprocess it."""
    # Load the red wine quality dataset from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    try:
        df = pd.read_csv(url, sep=";")
        print(f"✅ Dataset loaded from UCI repository: {df.shape}")
    except Exception:
        # Fallback: create the dataset URL for Kaggle-style loading
        print("⚠️  Could not load from UCI. Trying alternate source...")
        url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/winequality-red.csv"
        df = pd.read_csv(url, sep=";")
        print(f"✅ Dataset loaded from alternate source: {df.shape}")

    print(f"\nDataset Info:")
    print(f"  Rows: {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Features: {list(df.columns)}")
    print(f"\nTarget distribution (quality):")
    print(df['quality'].value_counts().sort_index())

    # Binarize the target: quality >= 7 is "Good" (1), else "Bad" (0)
    df['target'] = (df['quality'] >= 7).astype(int)
    print(f"\nBinarized target distribution:")
    print(f"  Bad (0):  {(df['target'] == 0).sum()}")
    print(f"  Good (1): {(df['target'] == 1).sum()}")

    # Separate features and target
    X = df.drop(['quality', 'target'], axis=1)
    y = df['target']

    return X, y, df


def train_and_evaluate():
    """Train all 5 models and evaluate them."""
    # Load data
    X, y, df = load_and_preprocess_data()

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    joblib.dump(scaler, os.path.join(script_dir, 'scaler.pkl'))
    print("\n✅ Scaler saved to model/scaler.pkl")

    # Save test data with target for the Streamlit app
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df['target'] = y_test.values
    test_df.to_csv(os.path.join(project_dir, 'test_data.csv'), index=False)
    print(f"✅ Test data saved to test_data.csv ({test_df.shape[0]} rows)")

    # Save feature names
    joblib.dump(list(X.columns), os.path.join(script_dir, 'feature_names.pkl'))

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    # Results storage
    results = []

    print("\n" + "=" * 80)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 80)

    for name, model in models.items():
        print(f"\n{'─' * 60}")
        print(f"Training: {name}")
        print(f"{'─' * 60}")

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Predict probabilities for AUC
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = model.decision_function(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall_val = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            'Model': name,
            'Accuracy': round(accuracy, 4),
            'AUC': round(auc, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall_val, 4),
            'F1 Score': round(f1, 4),
            'MCC': round(mcc, 4),
        })

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  AUC:       {auc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall_val:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  MCC:       {mcc:.4f}")

        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Bad', 'Good']))

        print(f"  Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"    {cm}")

        # Save model
        model_filename = name.lower().replace(' ', '_').replace('-', '_') + '.pkl'
        joblib.dump(model, os.path.join(script_dir, model_filename))
        print(f"  ✅ Model saved to model/{model_filename}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Save results
    results_df.to_csv(os.path.join(script_dir, 'results.csv'), index=False)
    print("\n✅ Results saved to model/results.csv")

    # Find winner
    winner_idx = results_df['F1 Score'].idxmax()
    print(f"\n🏆 Overall Winner: {results_df.loc[winner_idx, 'Model']} (Best F1 Score: {results_df.loc[winner_idx, 'F1 Score']})")

    return results_df


if __name__ == '__main__':
    results = train_and_evaluate()
