import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

model = None
scaler = None
feature_names = None
feature_info = None

def load_artifacts():
    global model, scaler, feature_names, feature_info
    if model is None:
        model = keras.models.load_model(os.path.join(MODELS_DIR, "nn_zero_to_sixty.keras"))
        with open(os.path.join(MODELS_DIR, "nn_scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "rb") as f:
            feature_names = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "feature_info.pkl"), "rb") as f:
            feature_info = pickle.load(f)