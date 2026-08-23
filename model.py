"""
model.py
Loads the training data, trains the Random Forest fraud model,
and exposes it along with feature importances for use elsewhere.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"


def load_data():
    df = pd.read_csv(DATA_URL)
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y


def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    )
    model.fit(X_train, y_train)

    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    return model, X_train, X_test, y_train, y_test, importances


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test, importances = train_model()
    print("Model trained successfully.")
    print("Top features:\n", importances.head(5))
