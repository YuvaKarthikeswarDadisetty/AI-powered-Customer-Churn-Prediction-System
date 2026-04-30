import matplotlib.pyplot as plt
import seaborn as sns
import os

def basic_info(df):
    print("\n📌 First 5 Rows:")
    print(df.head())

    print("\n📌 Dataset Shape:")
    print(df.shape)

    print("\n📌 Columns:")
    print(df.columns)

    print("\n📌 Info:")
    print(df.info())

    print("\n📌 Summary:")
    print(df.describe())

def check_missing_values(df):
    print("\n📌 Missing Values:")
    print(df.isnull().sum())

def plot_churn_distribution(df):
    os.makedirs("outputs", exist_ok=True)

    plt.figure()
    sns.countplot(x="Churn", data=df)
    plt.title("Churn Distribution")

    plt.savefig("outputs/churn_distribution.png")
    print("📊 Churn distribution plot saved")
    plt.close()