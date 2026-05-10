import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("processed_dataset.csv")

TARGET_COLUMN = "Loan_Status"

X = df.drop(TARGET_COLUMN, axis=1)

y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# MLFLOW AUTOLOG
# =========================================================

mlflow.set_experiment("Loan Prediction Experiment")

mlflow.sklearn.autolog()

# =========================================================
# TRAIN MODEL
# =========================================================

with mlflow.start_run():

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("Accuracy:", accuracy)

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(model, "artifacts/model.pkl")

    print("Model artifact saved.")