import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

model = None
scaler = None
feature_names = None
feature_info = None

def load_artifacts():
    global model, scaler, feature_names, feature_info
    if model is None:
        import keras
        model = keras.models.load_model(os.path.join(MODELS_DIR, "nn_zero_to_sixty.keras"))
        with open(os.path.join(MODELS_DIR, "nn_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "rb") as f:
            feature_names = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "feature_info.pkl"), "rb") as f:
            feature_info = pickle.load(f)

app = FastAPI(title="Supercar 0-100 Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CarInput(BaseModel):
    year: int
    horsepower: float
    engine_size: float
    torque: float
    weight: float
    drivetrain_rwd: int
    transmission_dct: int

@app.get("/")
def health():
    return {"status": "ok", "model": "nn_zero_to_sixty.keras"}

@app.post("/predict")
def predict(car: CarInput):
    load_artifacts()
    try:
        power_to_weight = car.horsepower / car.weight
        torque_to_weight = car.torque / car.weight

        raw = {
            "year": car.year,
            "horsepower": car.horsepower,
            "weight": car.weight,
            "torque": car.torque,
            "power_to_weight": power_to_weight,
            "torque_to_weight": torque_to_weight,
            "drivetrain_rwd": car.drivetrain_rwd,
            "transmission_dct": car.transmission_dct,
        }

        X = np.array([[raw[f] for f in feature_names]], dtype=np.float32)
        X_scaled = scaler.transform(X)

        prediction = float(model.predict(X_scaled, verbose=0)[0][0])
        prediction = max(0.0, round(prediction, 2))

        return {
            "prediction": prediction,
            "unit": "seconds",
            "power_weight_ratio": round(power_to_weight, 4),
            "torque_weight_ratio": round(torque_to_weight, 4),
            "estimated_weight": car.weight,
            "features_used": {
                "year": car.year,
                "horsepower": car.horsepower,
                "engine_size": car.engine_size,
                "torque": car.torque,
                "weight": car.weight,
                "weight_estimated": False,
                "power_weight": round(power_to_weight, 4),
                "torque_weight": round(torque_to_weight, 4),
                "drivetrain_rwd": bool(car.drivetrain_rwd),
                "transmission_dct": bool(car.transmission_dct),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))