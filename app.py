"""Task 5: FastAPI model-serving application."""
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
if not MODEL_PATH.exists():
    raise RuntimeError("model.pkl not found. Run `python train_model.py` first.")
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="REST API serving the trained Titanic classification pipeline.",
    version="1.0.0"
)

class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class: 1, 2, or 3")
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0, le=100)
    SibSp: int = Field(..., ge=0, le=20)
    Parch: int = Field(..., ge=0, le=20)
    Fare: float = Field(..., ge=0)
    Embarked: Literal["C", "Q", "S"] | None = None

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(passenger: Passenger):
    data = pd.DataFrame([passenger.model_dump()])[[
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
    ]]
    prediction = int(model.predict(data)[0])
    return {
        "prediction": prediction,
        "label": "Survived" if prediction == 1 else "Did not survive"
    }
