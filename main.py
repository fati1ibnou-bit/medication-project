import os
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# 1 INITIALISATION API

app = FastAPI(
    title="Medication Demand Prediction & Restocking API",
    description="API intelligente pour prédire la demande et gérer le stock pharmaceutique",
    version="1.0.0"
)

# 2 CHARGEMENT DU MODÈLE

MODEL_PATH = "model/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl introuvable. Lance train.py d'abord.")

model = joblib.load(MODEL_PATH)


# 3 SCHEMAS
class PredictionInput(BaseModel):
    medication_id: int
    rolling_7: float
    rolling_30: float
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    current_stock: float = 0



# 4 HOME

@app.get("/")
def home():
    return {
        "status": "API running",
        "message": "Medication Demand Prediction System 🚀",
        "docs": "/docs"
    }

# 5 PREDICTION

@app.post("/predict")
def predict(data: PredictionInput):

    try:
        x = np.array([[
            data.rolling_7,
            data.rolling_30,
            data.day_of_week,
            data.month
        ]])

        prediction = model.predict(x)[0]

        return {
            "medication_id": data.medication_id,
            "predicted_demand": round(float(max(0, prediction)), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# 6 RECOMMANDATION STOCK

@app.post("/recommend")
def recommend(data: PredictionInput):

    try:
        x = np.array([[
            data.rolling_7,
            data.rolling_30,
            data.day_of_week,
            data.month
        ]])

        prediction = max(0, float(model.predict(x)[0]))

        gap = prediction - data.current_stock

        if gap <= 0:
            status = "STOCK SUFFISANT"
        elif gap <= 5:
            status = "STOCK MOYEN"
        else:
            status = "URGENT RESTOCK"

        return {
            "medication_id": data.medication_id,
            "predicted_demand": round(prediction, 2),
            "current_stock": data.current_stock,
            "gap": round(gap, 2),
            "status": status
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))