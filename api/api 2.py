import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import keras

#  Load model & preprocessing artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

model = keras.models.load_model(os.path.join(MODELS_DIR, "nn_zero_to_sixty.keras"))

with open(os.path.join(MODELS_DIR, "nn_scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

with open(os.path.join(MODELS_DIR, "feature_info.pkl"), "rb") as f:
    feature_info = pickle.load(f)

#  App 
app = FastAPI(title="Supercar 0-100 Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Netlify URL after testing
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Request schema 
class CarInput(BaseModel):
    year: int
    horsepower: float
    weight_lbs: float
    torque: float
    drivetrain_rwd: int    # 1 = RWD, 0 = AWD
    transmission_dct: int  # 1 = DCT, 0 = Automatic

#  Endpoints 
@app.get("/")
def health():
    return {"status": "ok", "model": "nn_zero_to_sixty.keras"}


@app.post("/predict")
def predict(car: CarInput):
    try:
        # Derived physics features — must match training pipeline exactly
        power_to_weight = car.horsepower / car.weight_lbs
        torque_to_weight = car.torque / car.weight_lbs

        raw = {
            "year": car.year,
            "horsepower": car.horsepower,
            "weight": car.weight_lbs,
            "torque": car.torque,
            "power_to_weight": power_to_weight,
            "torque_to_weight": torque_to_weight,
            "drivetrain_rwd": car.drivetrain_rwd,
            "transmission_dct": car.transmission_dct,
        }

        # Align to exact feature order used during training
        X = np.array([[raw[f] for f in feature_names]], dtype=np.float32)
        X_scaled = scaler.transform(X)

        prediction = float(model.predict(X_scaled, verbose=0)[0][0])
        prediction = max(0.0, round(prediction, 2))

        return {
            "predicted_0_100_seconds": prediction,
            "inputs": car.model_dump(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))