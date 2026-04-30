import joblib
import pandas as pd

def load_pipeline():
    pipeline = joblib.load("models/churn_pipeline.pkl")
    return pipeline

def predict_new_customer(input_data: dict):
    pipeline = load_pipeline()

    df = pd.DataFrame([input_data])

    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    return prediction, probability