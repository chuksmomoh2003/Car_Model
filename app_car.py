from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import cloudpickle as cp

# Define the feature engineering function so that it's available when loading the pipeline
def feature_engineering(X):
    current_year = 2024
    
    # Calculate car_age and assign it back
    car_age = current_year - X['year']
    X['car_age'] = car_age
    
    # Calculate mileage per year and assign it back
    mileage_per_year = X['mileage'] / car_age
    X['mileage_per_year'] = mileage_per_year
    
    # Calculate fuel efficiency per engine size and handle infinite/NaN values
    fuel_efficiency = X['mpg'] / X['engineSize']
    fuel_efficiency.replace([np.inf, -np.inf], np.nan, inplace=True)
    fuel_efficiency.fillna(fuel_efficiency.median(), inplace=True)
    X['fuel_efficiency_per_engine_size'] = fuel_efficiency
    
    return X

# Load the saved pipeline (which includes both preprocessing and the model) using cloudpickle
with open('car_price_pipeline.pkl', 'rb') as f:
    pipeline = cp.load(f)

# Initialize FastAPI
app = FastAPI()

# Define input class for the request body (original variables)
class MyInput(BaseModel):
    model: str
    year: int
    transmission: str
    mileage: int
    fuelType: str
    tax: int
    mpg: float
    engineSize: float

@app.post('/predict/')
async def predict(input: MyInput):
    # Convert the input data to a DataFrame
    data = input.dict()
    data_ = pd.DataFrame([data])

    # Apply feature engineering
    data_ = feature_engineering(data_)

    # Use the pipeline to preprocess the data and make predictions
    prediction = pipeline.predict(data_)[0]

    return {
        'prediction': prediction
    }



