"""Task 3: Predict 20 unseen records together and save the results."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
UNSEEN_PATH = ROOT / "data" / "unseen_passengers.csv"
OUTPUT_PATH = ROOT / "outputs" / "batch_predictions.csv"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def main():
    if not MODEL_PATH.exists() or not UNSEEN_PATH.exists():
        raise FileNotFoundError("Run `python train_model.py` first.")
    model = joblib.load(MODEL_PATH)
    records = pd.read_csv(UNSEEN_PATH).head(20).copy()
    if len(records) < 20:
        raise ValueError(f"Need 20 unseen records, but found {len(records)}.")
    predictions = model.predict(records[FEATURES]).astype(int)
    records["Predicted_Survived"] = predictions
    records["Prediction_Label"] = records["Predicted_Survived"].map(
        {1: "Survived", 0: "Did not survive"}
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(OUTPUT_PATH, index=False)
    print("BATCH INFERENCE")
    print(records.to_string(index=False))
    print(f"\nSaved {len(records)} predictions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
