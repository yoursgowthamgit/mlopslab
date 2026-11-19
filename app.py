
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris KNN Prediction API")

# Load trained model
model = joblib.load("models/knn_model_v1.pkl")

# Prediction counter
total_predictions = 0

species_names = [
    "Iris-setosa",
    "Iris-versicolor",
    "Iris-virginica"
]

class IrisInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/model-info")
def model_info():
    return {
        "model": "KNN",
        "version": "V1",
        "n_neighbors": 5,
        "features": 5
    }

@app.post("/predict")
def predict(data: IrisInput):
    global total_predictions

    petal_area = data.PetalLengthCm * data.PetalWidthCm

    features = np.array([[
        data.SepalLengthCm,
        data.SepalWidthCm,
        data.PetalLengthCm,
        data.PetalWidthCm,
        petal_area
    ]])

    prediction = model.predict(features)[0]
    total_predictions += 1

    return {
        "predicted_species": species_names[int(prediction)],
        "prediction_number": total_predictions
    }

@app.get("/metrics")
def metrics():
    return {
        "total_predictions": total_predictions
    }
