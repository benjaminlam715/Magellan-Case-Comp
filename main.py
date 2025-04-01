import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression

stock_files = ["Starbucks.csv", "AMD.csv", "amazon.csv", "Cisco.csv", "msft.csv"]
predictions = {}

for file in stock_files:
    try:
        df = pd.read_csv(file)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Close"] = df["Close/Last"].str.replace("$", "", regex=False).astype(float)
        df = df.sort_values("Date").reset_index(drop=True)
        df = df[df["Date"] >= df["Date"].max() - pd.DateOffset(years=5)].copy()
        df["day_number"] = np.arange(len(df))

        X = df[["day_number"]].values
        y = df["Close"].values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)

        future_days = np.arange(len(df), len(df) + 365).reshape(-1, 1)
        future_preds = model.predict(future_days)

        last_date = df["Date"].iloc[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(365)]

        predictions[file.replace(".csv", "").upper()] = (future_dates, future_preds.flatten())
    except Exception as e:
        print(f"Error with {file}: {e}")

plt.figure(figsize=(14, 7))
for company, (dates, preds) in predictions.items():
    plt.plot(dates, preds, label=company, linewidth=2)

plt.title("1-Year Stock Price Forecast — 5 Company Comparison")
plt.xlabel("Date")
plt.ylabel("Predicted Close Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()