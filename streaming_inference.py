"""Task 4: Simulate streaming inference, one record every second."""
from pathlib import Path
import time
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
    records = pd.read_csv(UNSEEN_PATH).head(10)
    if len(records) < 10:
        raise ValueError(f"Need 10 unseen records, but found {len(records)}.")
    print("STREAMING INFERENCE (one passenger per second)")
    for _, row in records.iterrows():
        passenger = row[FEATURES].to_frame().T
        prediction = int(model.predict(passenger)[0])
        label = "Survived" if prediction == 1 else "Did not survive"
        print(f"PassengerId={row['PassengerId']}: prediction={prediction} ({label})")
        time.sleep(1)

if __name__ == "__main__":
    main()
