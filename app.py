import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# load model once (important!)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    features: list[float]


@app.get("/")
def health():
    return {"status": "model is running"}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"prediction": prediction[0]}

