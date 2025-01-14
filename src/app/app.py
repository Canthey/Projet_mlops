from fastapi import FastAPI
import mlflow
from pydantic import BaseModel
import numpy as np

model = mlflow.sklearn.load_model('runs:/4cdef354fc8b47398ca8ed34e5949bde/model')

app = FastAPI(
    title="Projet MLOPS Arnaud Rougier",
    description="Cette API permet de determiner le prix d'une maison en californie grace au modele entrainé sur le jeu de données California Housing",
    version="Description du projet"
)

@app.get("/")
def read_root():
    return {"message": "API Mlops Kenzée Aboustait"}

class HouseInformation(BaseModel):
    med_inc: float
    house_age: int
    ave_rooms: float
    ave_bedrooms: float
    population: int
    ave_occupation: float
    latitude: float
    longitude: float

@app.post("/predict")
def predict_value(house: HouseInformation):
    score = model.predict([
        [house.med_inc,
         house.house_age,
         house.ave_rooms,
         house.ave_bedrooms,
         house.population,
         house.ave_occupation,
         house.latitude,
         house.longitude]])
    return {"score": score[0]}