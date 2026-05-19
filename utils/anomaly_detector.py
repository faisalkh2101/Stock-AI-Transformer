import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    """
    Uses Unsupervised Machine Learning (Isolation Forest) to detect
    anomalous trading days based on price volatility and volume spikes.
    """
    try:
        data = df.copy()
        
        # 1. Feature Engineering: Calculate daily percentage changes
        data['Price_Change'] = data['Close'].pct_change() * 100
        data['Volume_Change'] = data['Volume'].pct_change() * 100

        # Drop NaN values for training
        train_data = data.dropna(subset=['Price_Change', 'Volume_Change'])

        if len(train_data) < 50:
            return False, "Not enough historical data for anomaly detection."

        # 2. Train the Unsupervised Model
        # contamination=0.02 means we expect only the top 2% most extreme days to be anomalies
        model = IsolationForest(contamination=0.02, random_state=42)
        features = train_data[['Price_Change', 'Volume_Change']]
        
        # Predict: -1 is an anomaly, 1 is normal
        train_data['Anomaly'] = model.fit_predict(features)

        # 3. Check the most recent trading day
        latest_data = train_data.iloc[-1]
        is_anomaly = latest_data['Anomaly'] == -1

        # 4. Generate the Explainable AI (XAI) reason
        reason = ""
        if is_anomaly:
            p_change = latest_data['Price_Change']
            v_change = latest_data['Volume_Change']
            
            if p_change < -4:
                reason = f"Flash Crash Warning: Price dropped severely by {p_change:.2f}%."
            elif p_change > 4:
                reason = f"Extreme Buying Surge: Price spiked abnormally by {p_change:.2f}%."
            elif v_change > 150:
                reason = f"Volume Anomaly: Trading volume spiked by {v_change:.2f}% compared to historical norms."
            else:
                reason = "Unusual mathematical combination of price action and volume detected by Isolation Forest."

        return is_anomaly, reason

    except Exception as e:
        print(f"Anomaly Detection Error: {e}")
        return False, "Error running detection."