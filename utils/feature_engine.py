import pandas as pd
import numpy as np

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(df):
    data = df.copy()
    data['SMA_50'] = calculate_sma(data, 50)
    data['SMA_200'] = calculate_sma(data, 200)
    data['RSI'] = calculate_rsi(data, 14)
    
    latest_row = data.iloc[-1]
    
    if pd.isna(latest_row['SMA_50']) or pd.isna(latest_row['RSI']):
         return "INSUFFICIENT DATA", "We need more historical trading days to calculate signals."
    
    signal = "HOLD"
    reason = "No strong momentum detected. Market is ranging."
    
    rsi = latest_row['RSI']
    sma50 = latest_row['SMA_50']
    sma200 = latest_row['SMA_200']
    
    if rsi < 30:
        signal = "STRONG BUY"
        reason = f"RSI is {rsi:.1f} (Oversold). The stock is heavily discounted."
    elif rsi > 70:
        signal = "STRONG SELL"
        reason = f"RSI is {rsi:.1f} (Overbought). The stock is trading at a premium."
    elif pd.notna(sma200):
        if sma50 > sma200:
            signal = "BUY"
            reason = "Golden Cross: Short-term momentum is higher than long-term trend."
        elif sma50 < sma200:
            signal = "SELL"
            reason = "Death Cross: Short-term momentum is dropping below long-term trend."
            
    return signal, reason