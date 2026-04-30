import pandas as pd

def clean_data(df):
    # Drop customerID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 🔥 FIX: Convert target variable
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    print("✅ Data cleaning completed")
    return df