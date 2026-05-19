import yfinance as yf
import pandas as pd
import streamlit as st

TOP_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC"
]

@st.cache_data(ttl=3600)
def fetch_dashboard_data():
    """Fetches the latest prices and calculates 24h changes for the home page."""
    try:
        tickers = yf.Tickers(" ".join(TOP_TICKERS))
        stock_stats = []
        for ticker in TOP_TICKERS:
            hist = tickers.tickers[ticker].history(period="5d")
            
            # CRITICAL FIX 1: Drop any rows where Yahoo Finance returned NaN for the Close price
            hist = hist.dropna(subset=['Close'])
            
            if len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                pct_change = ((current_price - prev_price) / prev_price) * 100
                stock_stats.append({
                    "Ticker": ticker,
                    "Price": round(current_price, 2),
                    "Change": round(pct_change, 2)
                })
        df_stats = pd.DataFrame(stock_stats)
        if not df_stats.empty:
            gainers = df_stats.sort_values(by="Change", ascending=False).head(5)
            losers = df_stats.sort_values(by="Change", ascending=True).head(5)
            return df_stats, gainers, losers
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data
def get_historical_data(ticker, start_date, end_date):
    """Fetches full historical data for the Detailed View and ML Training."""
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.reset_index(inplace=True)
    
    # CRITICAL FIX 2: Drop the empty/incomplete rows Yahoo Finance appends
    # This prevents the detailed view and Transformer model from breaking
    if not df.empty and 'Close' in df.columns:
        df = df.dropna(subset=['Close'])
        
    return df

def get_stock_info(ticker):
    """Fetches simple 52-week data for the quick view."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
        "previous_close": info.get("previousClose", "N/A"),
        "volume": info.get("volume", "N/A")
    }

@st.cache_data(ttl=86400) 
def get_company_fundamentals(ticker):
    """Fetches deep financial metrics and Wall Street analyst ratings."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        def format_large_number(num):
            if not num or num == "N/A": return "N/A"
            if num >= 1_000_000_000_000: return f"{num/1_000_000_000_000:.2f} Trillion"
            if num >= 1_000_000_000: return f"{num/1_000_000_000:.2f} Billion"
            if num >= 1_000_000: return f"{num/1_000_000:.2f} Million"
            return str(num)

        fundamentals = {
            "Market Cap": format_large_number(info.get("marketCap", "N/A")),
            "P/E Ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else "N/A",
            "Profit Margin": f"{round(info.get('profitMargins', 0) * 100, 2)}%" if info.get("profitMargins") else "N/A",
            "Dividend Yield": f"{round(info.get('dividendYield', 0) * 100, 2)}%" if info.get("dividendYield") else "N/A",
            "Analyst Recommendation": str(info.get("recommendationKey", "N/A")).replace("_", " ").title(),
            "Target Price": info.get("targetMeanPrice", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Summary": info.get("longBusinessSummary", "No company description available.")
        }
        return fundamentals
    except Exception as e:
        print(f"Fundamentals Error: {e}")
        return None