import joblib
import os
from src.pipeline import build_pipeline

def train_pipeline(df):
    pipeline = build_pipeline(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    pipeline.fit(X, y)

    print("✅ Pipeline trained successfully")
    return pipeline

def save_pipeline(pipeline):
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/churn_pipeline.pkl")
    print("💾 Pipeline saved successfully")