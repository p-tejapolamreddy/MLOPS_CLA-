"""Task 2: Predict one unseen passenger record."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
UNSEEN_PATH = ROOT / "data" / "unseen_passengers.csv"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def main():
    if not MODEL_PATH.exists() or not UNSEEN_PATH.exists():
        raise FileNotFoundError("Run `python train_model.py` first.")
    model = joblib.load(MODEL_PATH)
    records = pd.read_csv(UNSEEN_PATH)
    if records.empty:
        raise ValueError("No unseen records found.")
    passenger = records.iloc[[0]][FEATURES]
    prediction = int(model.predict(passenger)[0])
    label = "Survived" if prediction == 1 else "Did not survive"
    print("ONLINE INFERENCE")
    print("Passenger details:")
    print(passenger.to_string(index=False))
    print(f"Prediction: {prediction} ({label})")

if __name__ == "__main__":
    main()
