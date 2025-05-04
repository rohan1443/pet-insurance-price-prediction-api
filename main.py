# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle                                  # For (de)serializing the model :contentReference[oaicite:0]{index=0}
import numpy as np
import pandas as pd

# 1. Initialize FastAPI app and enable CORS for React/Vite frontend :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}
app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],   # match your frontend origin
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                        # for dev only; tighten in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],   # explicitly include OPTIONS :contentReference[oaicite:0]{index=0}
    allow_headers=["*"],
)


# 2. Load the pickled model once at startup :contentReference[oaicite:3]{index=3}
with open("./machine_learning_models/pet_insurance_model_5000.pkl", "rb") as f:
    model = pickle.load(f)

# 3. Pydantic model for incoming request body :contentReference[oaicite:4]{index=4}
class DogFeatures(BaseModel):
    age: float
    breed: str
    avg_daily_steps: int
    avg_resting_hr: int
    avg_daily_sleep: float
    meals_per_day: int
    activity_level: str
    health_events: int

# 4. The exact feature columns order your model was trained on
COLUMNS = ['Age (Years)',
 'Avg Daily Steps',
 'Avg Resting Heart Rate (bpm)',
 'Avg Daily Sleep (Hours)',
 'Meals per Day',
 'Health Events (Last Year)',
 'Activity_High',
 'Activity_Low',
 'Activity_Med',
 'Breed_Australian Shepherd',
 'Breed_Beagle',
 'Breed_Border Collie',
 'Breed_Boxer',
 'Breed_Bulldog',
 'Breed_Cavalier King Charles Spaniel',
 'Breed_Chihuahua',
 'Breed_Cocker Spaniel',
 'Breed_Dachshund',
 'Breed_Dalmatian',
 'Breed_French Bulldog',
 'Breed_German Shepherd',
 'Breed_Golden Retriever',
 'Breed_Great Dane',
 'Breed_Jack Russell Terrier',
 'Breed_Labrador Retriever',
 'Breed_Pomeranian',
 'Breed_Pug',
 'Breed_Rottweiler',
 'Breed_Siberian Husky']

@app.post("/predict")
async def predict(features: DogFeatures):
    """
    1. Parse and validate input via Pydantic.
    2. Build a single-row DataFrame with one-hot and numeric features.
    3. Reorder columns, convert to NumPy array, and predict.
    4. Return rounded premium.
    """
    try:
        # 5. Numeric features mapping
        num_feats = {
            'Age (Years)': features.age,
            'Avg Daily Steps': features.avg_daily_steps,
            'Avg Resting Heart Rate (bpm)': features.avg_resting_hr,
            'Avg Daily Sleep (Hours)': features.avg_daily_sleep,
            'Meals per Day': features.meals_per_day,
            'Health Events (Last Year)': features.health_events
        }

        # 6. One-hot encode activity_level :contentReference[oaicite:5]{index=5}
        act = features.activity_level.lower()
        activity_feats = {
            'Activity_High': 1 if act == 'high' else 0,
            'Activity_Low': 1 if act == 'low' else 0,
            'Activity_Medium': 1 if act in ('med', 'medium') else 0
        }

        # 7. One-hot encode breed :contentReference[oaicite:6]{index=6}
        breed = features.breed
        breed_feats = {col: (1 if col == f"Breed_{breed}" else 0) 
                       for col in COLUMNS if col.startswith("Breed_")}
        
        
        # breed_feats = {col: 0 for col in COLUMNS if col.startswith("Breed_")}  # Initialize all breeds to 0
        # breed_column = f"Breed_{features.breed}"  # Correct column name for the breed
        # if breed_column in breed_feats:
        #     breed_feats[breed_column] = 1  # Set the selected breed to 1


        # 8. Combine all features into one dict
        data = {**num_feats, **activity_feats, **breed_feats}

        # 9. Create DataFrame and ensure correct column order
        df = pd.DataFrame([data], columns=COLUMNS)  # single-row DF :contentReference[oaicite:7]{index=7}

        # 10. Convert to NumPy array and predict
        # X = df.to_numpy()  # shape (1, n_features) :contentReference[oaicite:8]{index=8}
        # pred = model.predict(df)[0]
        
        pred = model.predict(df)[0]

        return {"premium": round(float(pred), 2)}

    except Exception as e:
        # 11. Return HTTP 500 on unexpected errors
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
