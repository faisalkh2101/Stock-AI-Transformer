import pandas as pd
import numpy as np
from utils.feature_engine import calculate_sma, calculate_rsi

def run_backtest(df, initial_capital=10000):
    """
    Simulates trading the stock using the Baseline Algorithm (SMA Crossovers & RSI)
    over the provided historical data.
    """
    try:
        # 1. Prepare the historical indicators
        data = df.copy()
        data['SMA_50'] = calculate_sma(data, 50)
        data['SMA_200'] = calculate_sma(data, 200)
        data['RSI'] = calculate_rsi(data, 14)
        
        cash = initial_capital
        shares = 0
        trade_log = []
        portfolio_history = []

        # 2. Run the simulation day-by-day
        for i in range(1, len(data)):
            date = data['Date'].iloc[i]
            price = data['Close'].iloc[i]
            rsi = data['RSI'].iloc[i]
            sma50 = data['SMA_50'].iloc[i]
            sma200 = data['SMA_200'].iloc[i]
            
            prev_sma50 = data['SMA_50'].iloc[i-1]
            prev_sma200 = data['SMA_200'].iloc[i-1]

            # Generate historical signal
            signal = "HOLD"
            if pd.notna(rsi):
                if rsi < 30:
                    signal = "BUY"
                elif rsi > 70:
                    signal = "SELL"
            
            if pd.notna(sma200) and pd.notna(prev_sma200) and signal == "HOLD":
                if sma50 > sma200 and prev_sma50 <= prev_sma200:
                    signal = "BUY" # Golden Cross
                elif sma50 < sma200 and prev_sma50 >= prev_sma200:
                    signal = "SELL" # Death Cross

            # Execute Trade
            if signal == "BUY" and cash >= price:
                shares_bought = int(cash // price)
                if shares_bought > 0:
                    cash -= shares_bought * price
                    shares += shares_bought
                    trade_log.append({"Date": date, "Action": "🟢 BUY", "Price": price, "Shares": shares_bought, "Total Value": shares_bought * price})
            
            elif signal == "SELL" and shares > 0:
                sale_value = shares * price
                cash += sale_value
                trade_log.append({"Date": date, "Action": "🔴 SELL", "Price": price, "Shares": shares, "Total Value": sale_value})
                shares = 0
                
            # Record daily portfolio value
            current_value = cash + (shares * price)
            portfolio_history.append({"Date": date, "Portfolio Value": current_value})

        # 3. Calculate Final Metrics
        final_price = data['Close'].iloc[-1]
        final_value = cash + (shares * final_price)
        strategy_return = ((final_value - initial_capital) / initial_capital) * 100

        # Calculate Buy & Hold baseline (If we just bought on day 1 and did nothing)
        initial_price = data['Close'].iloc[0]
        bnh_shares = initial_capital / initial_price
        bnh_final_value = bnh_shares * final_price
        bnh_return = ((bnh_final_value - initial_capital) / initial_capital) * 100

        history_df = pd.DataFrame(portfolio_history)
        trades_df = pd.DataFrame(trade_log)

        return {
            "success": True,
            "final_value": final_value,
            "strategy_return": strategy_return,
            "bnh_return": bnh_return,
            "total_trades": len(trade_log),
            "history_df": history_df,
            "trades_df": trades_df
        }
    except Exception as e:
        print(f"Backtest Error: {e}")
        return {"success": False}