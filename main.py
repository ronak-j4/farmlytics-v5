from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import shap
import os
import requests
from dotenv import load_dotenv

# Import irrigation prediction logic
from irrigation_lstm_backend.predict_lstm import predict_irrigation

load_dotenv()

app = FastAPI(title="Farmlytics Crop Recommendation API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model
model_path = 'model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
else:
    model = None
    explainer = None
    print("Warning: model.pkl not found. Please run train_model.py first.")

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class WeatherCropInput(BaseModel):
    mode: str
    city: Optional[str] = None
    state: Optional[str] = None
    N: float
    P: float
    K: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    ph: float
    rainfall: Optional[float] = None

class IrrigationInput(BaseModel):
    location: str
    cropType: Optional[str] = None
    soilType: Optional[str] = None
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    soilPH: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None

@app.post("/predict")
def predict_crop(data: CropInput):
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Create DataFrame for prediction to match feature names
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_data = pd.DataFrame([{
            'N': data.N,
            'P': data.P,
            'K': data.K,
            'temperature': data.temperature,
            'humidity': data.humidity,
            'ph': data.ph,
            'rainfall': data.rainfall
        }])
        
        # 1. Prediction & Confidence Score
        prediction = model.predict(input_data)[0]
        predicted_crop = prediction.capitalize()
        
        probabilities = model.predict_proba(input_data)[0]
        class_index = list(model.classes_).index(prediction)
        confidence_score = round(float(probabilities[class_index]), 2)
        
        # 2. Global Feature Importance
        importances = model.feature_importances_
        feature_importance = {feat: round(float(imp), 2) for feat, imp in zip(features, importances)}
        
        # 3. SHAP Explainability (Local Feature Importance)
        shap_values = explainer.shap_values(input_data)
        
        # shap_values is a list of arrays (one per class) for classification
        # Get the shap values for the predicted class
        class_shap_values = shap_values[class_index][0]
        
        # Pair features with their shap values and sort by absolute impact
        feature_impacts = []
        for i, feat in enumerate(features):
            impact_val = class_shap_values[i]
            feature_impacts.append({
                "feature": feat,
                "impact": "positive" if impact_val > 0 else "negative",
                "abs_val": abs(impact_val)
            })
            
        # Sort by absolute impact descending and get top 3
        feature_impacts.sort(key=lambda x: x["abs_val"], reverse=True)
        top_influencing_factors = [
            {"feature": item["feature"], "impact": item["impact"]} 
            for item in feature_impacts[:3]
        ]
        
        crop_info = {
            "mango": {
                "effort": "High (Pruning, pest control, years to harvest)",
                "water": "Needs a dry spell to flower",
                "risk": "High humidity can cause fruit fungus",
                "return": "High long-term value"
            },
            "rice": {
                "effort": "High (Labor-intensive planting and harvesting)",
                "water": "Requires flooded conditions for most of the season",
                "risk": "Vulnerable to drought and water mismanagement",
                "return": "Moderate but stable staple crop"
            },
            "maize": {
                "effort": "Moderate (Mechanized farming possible)",
                "water": "Requires consistent moisture, especially during silking",
                "risk": "Susceptible to fall armyworm and drought",
                "return": "Moderate to high depending on market"
            },
            "chickpea": {
                "effort": "Low to Moderate",
                "water": "Drought tolerant, requires minimal irrigation",
                "risk": "Susceptible to pod borer and wilt",
                "return": "High value pulse crop"
            },
            "kidneybeans": {
                "effort": "Moderate",
                "water": "Requires well-distributed rainfall",
                "risk": "Sensitive to waterlogging and extreme heat",
                "return": "High value pulse crop"
            },
            "pigeonpeas": {
                "effort": "Low",
                "water": "Highly drought tolerant",
                "risk": "Susceptible to pod borers",
                "return": "Moderate to high"
            },
            "mothbeans": {
                "effort": "Low",
                "water": "Extremely drought tolerant",
                "risk": "Low risk, hardy crop",
                "return": "Moderate"
            },
            "mungbean": {
                "effort": "Low",
                "water": "Drought tolerant, short duration",
                "risk": "Susceptible to yellow mosaic virus",
                "return": "Moderate"
            },
            "blackgram": {
                "effort": "Low",
                "water": "Drought tolerant",
                "risk": "Susceptible to yellow mosaic virus",
                "return": "Moderate"
            },
            "lentil": {
                "effort": "Low",
                "water": "Requires minimal water",
                "risk": "Susceptible to wilt and rust",
                "return": "Moderate to high"
            },
            "pomegranate": {
                "effort": "High",
                "water": "Drought tolerant but needs regular watering for yield",
                "risk": "Susceptible to bacterial blight",
                "return": "Very high"
            },
            "banana": {
                "effort": "High",
                "water": "Requires high and consistent moisture",
                "risk": "Susceptible to Panama wilt and wind damage",
                "return": "High"
            },
            "grapes": {
                "effort": "Very High (Trellising, pruning, spraying)",
                "water": "Requires controlled irrigation",
                "risk": "Highly susceptible to fungal diseases",
                "return": "Very high"
            },
            "watermelon": {
                "effort": "Moderate",
                "water": "Requires high water during fruit development",
                "risk": "Susceptible to fruit fly and cracking",
                "return": "High in summer season"
            },
            "muskmelon": {
                "effort": "Moderate",
                "water": "Requires controlled watering",
                "risk": "Susceptible to powdery mildew",
                "return": "High"
            },
            "apple": {
                "effort": "High",
                "water": "Requires regular watering",
                "risk": "Susceptible to scab and hail damage",
                "return": "Very high"
            },
            "orange": {
                "effort": "High",
                "water": "Requires regular irrigation",
                "risk": "Susceptible to citrus greening and fruit drop",
                "return": "High"
            },
            "papaya": {
                "effort": "Moderate",
                "water": "Requires good drainage, sensitive to waterlogging",
                "risk": "Highly susceptible to papaya ringspot virus",
                "return": "High"
            },
            "coconut": {
                "effort": "Moderate",
                "water": "Requires high rainfall or regular irrigation",
                "risk": "Susceptible to rhinoceros beetle and lethal yellowing",
                "return": "High long-term value"
            },
            "cotton": {
                "effort": "High",
                "water": "Requires moderate water, dry spell for boll bursting",
                "risk": "Highly susceptible to bollworm",
                "return": "High cash crop"
            },
            "jute": {
                "effort": "High (Retting process is labor intensive)",
                "water": "Requires high rainfall and standing water for retting",
                "risk": "Price volatility",
                "return": "Moderate"
            },
            "coffee": {
                "effort": "High",
                "water": "Requires well-distributed rainfall",
                "risk": "Susceptible to coffee berry borer and rust",
                "return": "High export value"
            }
        }

        details = crop_info.get(prediction.lower(), {
            "effort": "Information not available for this crop.",
            "water": "Information not available for this crop.",
            "risk": "Information not available for this crop.",
            "return": "Information not available for this crop."
        })

        return {
            "predicted_crop": predicted_crop,
            "confidence_score": confidence_score,
            "feature_importance": feature_importance,
            "top_influencing_factors": top_influencing_factors,
            "details": details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-with-weather")
def predict_with_weather(data: WeatherCropInput):
    if model is None or explainer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    temperature = data.temperature
    humidity = data.humidity
    rainfall = data.rainfall
    weather_source = "Manual"

    if data.mode == "auto":
        if not data.city or not data.state:
            raise HTTPException(status_code=400, detail="City and state are required for auto mode.")
        
        api_key = os.getenv("WEATHER_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="WEATHER_API_KEY not configured in .env file.")
        
        try:
            query = f"{data.city},{data.state},India"
            url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={query}"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                error_msg = response.json().get('error', {}).get('message', 'Unknown error')
                raise HTTPException(status_code=400, detail=f"Failed to fetch weather data: {error_msg}")
            
            weather_data = response.json()
            temperature = weather_data["current"]["temp_c"]
            humidity = weather_data["current"]["humidity"]
            rainfall = weather_data["current"]["precip_mm"]
                
            weather_source = "WeatherAPI"
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=504, detail="Weather service request timed out.")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error connecting to weather service: {str(e)}")
    else:
        if temperature is None or humidity is None or rainfall is None:
            raise HTTPException(status_code=400, detail="Temperature, humidity, and rainfall are required for manual mode.")

    try:
        # Create DataFrame for prediction to match feature names
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_data = pd.DataFrame([{
            'N': data.N,
            'P': data.P,
            'K': data.K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': data.ph,
            'rainfall': rainfall
        }])
        
        # 1. Prediction & Confidence Score
        prediction = model.predict(input_data)[0]
        predicted_crop = prediction.capitalize()
        
        probabilities = model.predict_proba(input_data)[0]
        class_index = list(model.classes_).index(prediction)
        confidence_score = round(float(probabilities[class_index]), 2)
        
        return {
            "predicted_crop": predicted_crop,
            "confidence_score": confidence_score * 100,
            "weather_used": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall
            },
            "weather_source": weather_source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-irrigation")
def predict_irrigation_endpoint(data: IrrigationInput):
    """
    Predict irrigation requirement for next 5 days using LSTM model.
    """
    try:
        # Geocoding to get latitude and longitude from location string
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={data.location}&count=1"
        geo_response = requests.get(geo_url)
        if geo_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch location coordinates.")
            
        geo_data = geo_response.json()
        if not geo_data.get("results"):
            raise HTTPException(status_code=400, detail=f"Location '{data.location}' not found.")
            
        latitude = geo_data["results"][0]["latitude"]
        longitude = geo_data["results"][0]["longitude"]
        
        # Call the internal function for LSTM prediction
        irrigation_plan = predict_irrigation(latitude, longitude)
        
        # Return the exact JSON structure requested
        return irrigation_plan
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Crop Recommendation ML API is running!"}
