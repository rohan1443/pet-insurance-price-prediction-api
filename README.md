# pet-insurance-price-prediction-api

## Description
A simple API built to serve as a REST service using the pet_insurance_model as a regression model for the purpose of insurance premium price prediction based on payload as input

## How to run the api in local
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

## To test the API
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 2.0,
    "breed": "Beagle",
    "avg_daily_steps": 10000,
    "avg_resting_hr": 65,
    "avg_daily_sleep": 8.0,
    "meals_per_day": 3,
    "activity_level": "Med",
    "health_events": 0
  }'

## Deployment
https://railway.com (gives you a free 5$ credit under trial plan)
<img width="1494" alt="image" src="https://github.com/user-attachments/assets/347998b6-18e7-4115-a6d5-fbfe6793bde7" />

