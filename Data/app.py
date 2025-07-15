from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('player_pts_model_rf.joblib')

class InputData(BaseModel):
    avg_pts_l5: float
    avg_reb_l5: float
    avg_ast_l5: float
    opp_def_rating: float
    minutes: float

@app.post("/predict/points")
def predict(data: InputData):
    features = np.array([
        data.avg_pts_l5,
        data.avg_reb_l5,
        data.avg_ast_l5,
        data.opp_def_rating,
        data.minutes
    ]).reshape(1, -1)
    pred = model.predict(features)[0]
    return {"predicted_points": pred}