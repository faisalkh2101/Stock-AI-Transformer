import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import streamlit as st
import os

@st.cache_resource
def load_prediction_model():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'lstm_model.keras')
        scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
        
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_next_day(df_hist, model, scaler):
    try:
        last_60_days = df_hist['Close'].values[-60:]
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predicted_price_scaled = model.predict(X_test, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        
        return round(float(predicted_price[0][0]), 2)
    except Exception as e:
        print(f"Prediction Error: {e}")
        return None