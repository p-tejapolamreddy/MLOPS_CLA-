"""Task 1: Train and evaluate a Titanic survival classification pipeline.

Download train.csv from the Kaggle competition and place it at data/train.csv.
This script uses a stratified train/test split. The model is trained only on the
training portion; unseen_passengers.csv is created from held-out test rows.
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "model.pkl"
UNSEEN_PATH = ROOT / "data" / "unseen_passengers.csv"
METRICS_PATH = ROOT / "outputs" / "evaluation.txt"

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Download train.csv from "
            "https://www.kaggle.com/competitions/titanic/data and save it in data/train.csv."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + ["Survived"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = df[FEATURES].copy()
    y = df["Survived"].astype(int)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)

    # Held-out rows are unseen by the fitted model. Remove labels before inference.
    unseen = X_test.iloc[:20].copy()
    unseen.insert(0, "PassengerId", df.loc[X_test.index[:20], "PassengerId"].values)
    unseen.to_csv(UNSEEN_PATH, index=False)

    report = (
        f"Dataset rows: {len(df)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n"
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
        f"Classification report:\n{classification_report(y_test, predictions, zero_division=0)}"
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(report, encoding="utf-8")

    print("Model training completed.")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved held-out inference records: {UNSEEN_PATH}")
    print(f"Saved evaluation details: {METRICS_PATH}")

if __name__ == "__main__":
    main()
