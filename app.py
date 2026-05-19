import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta

# Import our backend engines
from utils.data_handler import fetch_dashboard_data, get_historical_data, get_stock_info, get_company_fundamentals
from utils.feature_engine import generate_signals, calculate_sma, calculate_rsi 
from utils.nlp_engine import get_news_sentiment 
from utils.backtester import run_backtest 
from utils.anomaly_detector import detect_anomalies 
from utils.transformer_engine import predict_with_transformer 

# 1. Page Configuration
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- PREMIUM CUSTOM CSS INJECTION ---
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    div[data-testid="metric-container"] {
        background-color: #1b2028; 
        border-radius: 10px; 
        padding: 15px; 
        border: 1px solid #2d333f; 
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2); 
    }
    
    div.stButton > button {
        min-height: 110px; 
        width: 100%;
        border-radius: 15px;
        background: linear-gradient(145deg, #1b2028, #13171c); 
        border: 1px solid #2d333f;
        box-shadow: 4px 4px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease; 
        white-space: pre-wrap; 
    }
    
    div.stButton > button:hover {
        border-color: #00E676;
        color: #00E676;
        transform: translateY(-5px); 
        box-shadow: 0px 8px 15px rgba(0, 230, 118, 0.2); 
    }
    
    div.stButton > button p {
        font-size: 1.15rem;
        font-weight: bold;
        line-height: 1.4;
    }
    
    @keyframes pulseAlert {
        0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    .anomaly-alert {
        animation: pulseAlert 2s infinite;
        border: 2px solid #ff4b4b;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'selected_stock' not in st.session_state: st.session_state.selected_stock = None
if 'search_query' not in st.session_state: st.session_state.search_query = ""

def go_back():
    st.session_state.selected_stock = None
    st.session_state.search_query = "" 

# --- TOP BAR: SEARCH ---
st.title("🦅 Market Prediction Dashboard")
st.text_input("🔍 Search for a Stock Ticker (e.g., AAPL, RELIANCE.NS, TCS.NS)", placeholder="Type ticker and press Enter...", key="search_query")

if st.session_state.search_query:
    st.session_state.selected_stock = st.session_state.search_query.upper()

# --- IF A STOCK IS SELECTED: SHOW DETAILED VIEW ---
if st.session_state.selected_stock:
    ticker = st.session_state.selected_stock
    st.button("← Back to Home Page", on_click=go_back)
    
    # --- ENTERPRISE DATA PIPELINE ---
    with st.spinner(f"Loading Enterprise Data Pipeline & Running AI Models for {ticker}..."):
        end_date = date.today()
        start_date = end_date - timedelta(days=730) 
        
        # 1. Fetch Market Data
        df_hist = get_historical_data(ticker, start_date, end_date)
        stock_info = get_stock_info(ticker)
        
        # 2. Add Indicators necessary for Transformer
        df_hist['SMA_50'] = calculate_sma(df_hist, 50).round(2)
        df_hist['SMA_200'] = calculate_sma(df_hist, 200).round(2)
        df_hist['RSI'] = calculate_rsi(df_hist, 14).round(2)
        
        # 3. Fetch Global NLP Sentiment BEFORE running price prediction
        df_news, avg_score, consensus = get_news_sentiment(ticker)
        
    if df_hist.empty:
        st.error("Could not fetch data for this ticker. Please check the spelling.")
    else:
        # --- EXPORT TO CSV ---
        csv_data = df_hist.to_csv(index=False).encode('utf-8')
        
        header_col, btn_col = st.columns([3, 1])
        with header_col:
            st.header(f"Detailed Analysis: {ticker}")
        with btn_col:
            st.write("") 
            st.download_button(
                label="📥 Download AI Report (CSV)",
                data=csv_data,
                file_name=f"{ticker}_Quantitative_Report.csv",
                mime="text/csv",
                use_container_width=True
            )

        current_price = df_hist['Close'].iloc[-1]
        currency = "₹" if ".NS" in ticker else "$"

        # --- 🚨 UNSUPERVISED ML: FLASH CRASH RADAR 🚨 ---
        is_anomaly, anomaly_reason = detect_anomalies(df_hist)
        if is_anomaly:
            st.markdown('<div class="anomaly-alert">', unsafe_allow_html=True)
            st.error(f"🚨 **UNSUPERVISED AI ALERT (Isolation Forest):** {anomaly_reason}")
            st.markdown('</div>', unsafe_allow_html=True)

        # --- CREATE TABS ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview & AI Forecast", 
            "📰 Live News & Sentiment", 
            "🏢 Company Fundamentals",
            "🔄 Strategy Backtesting"
        ])
        
        # ==========================================
        # TAB 1: CHART AND PREDICTIONS
        # ==========================================
        with tab1:
            # --- NEXT-GEN TRANSFORMER FORECAST ---
            st.subheader("🧠 Next-Gen AI Forecast (Multivariate Transformer)")
            predicted_price, attention_weights = predict_with_transformer(df_hist, avg_score)
            
            if predicted_price and attention_weights:
                expected_change = ((predicted_price - current_price) / current_price) * 100
                pred_color = "normal" if expected_change >= 0 else "inverse"
                
                t_col1, t_col2 = st.columns([1, 2])
                with t_col1:
                    st.metric("Predicted Next Close", f"{currency} {predicted_price:,.2f}", f"{expected_change:.2f}% (Expected)", delta_color=pred_color)
                
                with t_col2:
                    st.write("**Explainable AI: Model Attention Weights**")
                    # Visualizing the Transformer's Attention Layers
                    st.progress(int(attention_weights['Price Action & Momentum']), text=f"📈 Historical Price Momentum ({attention_weights['Price Action & Momentum']}%)")
                    st.progress(int(attention_weights['Live NLP News Sentiment']), text=f"📰 Live NLP News Sentiment ({attention_weights['Live NLP News Sentiment']}%)")
                    st.progress(int(attention_weights['Technical Indicators (RSI/Vol)']), text=f"📊 Technical Indicators & Volume ({attention_weights['Technical Indicators (RSI/Vol)']}%)")
            else:
                st.warning("Transformer model failed to generate a prediction due to insufficient data.")
                
            st.divider()

            # --- BASELINE ALGORITHMIC SIGNALS ---
            st.subheader("📊 Baseline Algorithmic Signals")
            signal, reason = generate_signals(df_hist)
            signal_color = "🟢" if "BUY" in signal else "🔴" if "SELL" in signal else "🟡"
            
            signal_col, m1, m2, m3, m4 = st.columns(5)
            signal_col.info(f"{signal_color} **Rule-Based Signal: {signal}**\n\n*Reason:* {reason}") 
            m1.metric("Current Price", f"{currency} {current_price:,.2f}")
            m2.metric("52-Week High", stock_info['52_week_high'])
            m3.metric("52-Week Low", stock_info['52_week_low'])
            m4.metric("Volume", f"{stock_info['volume']:,}" if stock_info['volume'] != "N/A" else "N/A")
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_hist['Date'], open=df_hist['Open'], high=df_hist['High'], low=df_hist['Low'], close=df_hist['Close'], name='Price'))
            fig.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['Close'].rolling(50).mean(), line=dict(color='orange', width=1.5), name='50-Day SMA'))
            fig.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['Close'].rolling(200).mean(), line=dict(color='#00E676', width=1.5), name='200-Day SMA')) 
            fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        # ==========================================
        # TAB 2: NLP NEWS SENTIMENT 
        # ==========================================
        with tab2:
            st.subheader(f"Live News & AI Sentiment Analysis for {ticker}")
            if df_news.empty:
                st.warning("No recent news articles found for this stock right now.")
            else:
                cons_color = "🟢" if "Bullish" in consensus else "🔴" if "Bearish" in consensus else "🟡"
                c1, c2 = st.columns(2)
                c1.metric("Market Sentiment Consensus", f"{cons_color} {consensus}")
                c2.metric("Average NLP Score", f"{avg_score:.2f}", help="Scores range from -1.0 to +1.0.")
                st.divider()
                st.write("### Recent Headlines Feed")
                for i, row in df_news.iterrows():
                    with st.container():
                        st.markdown(f"#### {row['Headline']}")
                        st.caption(f"**Publisher:** {row['Publisher']} &nbsp;&nbsp;|&nbsp;&nbsp; **AI Assessment:** {row['Sentiment']} &nbsp;&nbsp;|&nbsp;&nbsp; [Read Full Article]({row['Link']})")
                        st.write("---")

        # ==========================================
        # TAB 3: COMPANY FUNDAMENTALS 
        # ==========================================
        with tab3:
            st.subheader(f"Corporate Overview: {ticker}")
            with st.spinner("Fetching financial disclosures and analyst ratings..."):
                fundamentals = get_company_fundamentals(ticker)
                
            if not fundamentals: st.warning("Financial fundamentals are currently unavailable.")
            else:
                st.markdown(f"**Sector:** {fundamentals['Sector']} | **Industry:** {fundamentals['Industry']}")
                with st.expander("📖 Read Business Summary", expanded=True):
                    st.write(fundamentals['Summary'])
                st.divider()
                
                st.subheader("📊 Financial Health & Valuation")
                f_col1, f_col2, f_col3, f_col4 = st.columns(4)
                f_col1.metric("Market Capitalization", fundamentals['Market Cap'])
                f_col2.metric("P/E Ratio (Trailing)", fundamentals['P/E Ratio'])
                f_col3.metric("Profit Margin", fundamentals['Profit Margin'])
                f_col4.metric("Dividend Yield", fundamentals['Dividend Yield'])
                st.divider()
                
                st.subheader("👔 Wall Street Analyst Consensus")
                a_col1, a_col2 = st.columns(2)
                rec = fundamentals['Analyst Recommendation']
                st_rec = f"🟢 **{rec}**" if "Buy" in rec else f"🔴 **{rec}**" if "Sell" in rec or "Under" in rec else f"🟡 **{rec}**"
                a_col1.info(f"Consensus Rating: {st_rec}")
                
                target_price = fundamentals['Target Price']
                if target_price != "N/A":
                    target_diff = ((target_price - current_price) / current_price) * 100
                    a_col2.metric("Average 1yr Target Price", f"{currency} {target_price:,.2f}", f"{target_diff:.2f}% (Expected)")
                else: a_col2.metric("Average 1yr Target Price", "N/A")

        # ==========================================
        # TAB 4: ALGORITHMIC BACKTESTING
        # ==========================================
        with tab4:
            st.subheader(f"Quantitative Backtesting Engine: {ticker}")
            st.markdown("This engine simulates trading **2 years of historical data** using our Baseline Algorithm (RSI & 50/200-day Moving Averages) to prove the viability of the strategy.")
            
            col_input, col_empty = st.columns([1, 3])
            with col_input:
                initial_capital = st.number_input("Starting Capital", min_value=1000, max_value=1000000, value=10000, step=1000)
                
            if st.button("▶️ Run Simulation", use_container_width=True):
                with st.spinner("Running historical simulations..."):
                    results = run_backtest(df_hist, initial_capital)
                    
                if results["success"]:
                    st.divider()
                    st.subheader("📊 Simulation Results")
                    
                    strat_color = "normal" if results['strategy_return'] >= 0 else "inverse"
                    bnh_color = "normal" if results['bnh_return'] >= 0 else "inverse"
                    
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Initial Capital", f"{currency} {initial_capital:,.2f}")
                    r2.metric("Final Portfolio Value", f"{currency} {results['final_value']:,.2f}", f"{results['strategy_return']:.2f}%", delta_color=strat_color)
                    r3.metric("Buy & Hold Return", f"{results['bnh_return']:.2f}%", help="What you would have made if you just bought on Day 1 and held.", delta_color=bnh_color)
                    r4.metric("Total Trades Executed", results['total_trades'])
                    
                    if not results['history_df'].empty:
                        st.write("### Portfolio Equity Curve")
                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(x=results['history_df']['Date'], y=results['history_df']['Portfolio Value'], mode='lines', name='Strategy Equity', line=dict(color='#00E676', width=2)))
                        fig_eq.add_hline(y=initial_capital, line_dash="dot", line_color="gray", annotation_text="Starting Capital")
                        fig_eq.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Portfolio Value")
                        st.plotly_chart(fig_eq, use_container_width=True)
                    
                    st.write("### Historical Trade Log")
                    if not results['trades_df'].empty:
                        st.dataframe(results['trades_df'], use_container_width=True, hide_index=True)
                    else:
                        st.info("The algorithm did not find any strong buy/sell signals during this period. Your capital was held safely in cash.")
                else:
                    st.error("Simulation failed. Not enough historical data to generate moving averages.")

# --- IF NO STOCK IS SELECTED: SHOW HOME PAGE ---
else:
    with st.spinner("Fetching live market data from Yahoo Finance..."):
        df_stats, gainers, losers = fetch_dashboard_data()

    if df_stats.empty:
        st.error("Failed to load market data. Please check your internet connection or try again later.")
    else:
        st.subheader("🔥 Top Gainers (24h)")
        g_cols = st.columns(len(gainers))
        for i, (_, row) in enumerate(gainers.iterrows()):
            ticker = row['Ticker']
            price = row['Price']
            change = row['Change']
            currency = "₹" if ".NS" in ticker else "$"
            
            with g_cols[i]:
                if st.button(f"{ticker}\n{currency}{price} (+{change}%)", key=f"gain_{ticker}", use_container_width=True):
                    st.session_state.selected_stock = ticker
                    st.rerun()

        st.subheader("🩸 Top Losers (24h)")
        l_cols = st.columns(len(losers))
        for i, (_, row) in enumerate(losers.iterrows()):
            ticker = row['Ticker']
            price = row['Price']
            change = row['Change']
            currency = "₹" if ".NS" in ticker else "$"
            
            with l_cols[i]:
                if st.button(f"{ticker}\n{currency}{price} ({change}%)", key=f"lose_{ticker}", use_container_width=True):
                    st.session_state.selected_stock = ticker
                    st.rerun()
                    
        st.divider()

        st.subheader("🏢 Top Stocks (Click to Analyze)")
        grid_columns = st.columns(5)
        for i, (_, row) in enumerate(df_stats.iterrows()):
            ticker = row['Ticker']
            price = row['Price']
            change = row['Change']
            currency = "₹" if ".NS" in ticker else "$"
            with grid_columns[i % 5]:
                if st.button(f"{ticker}\n{currency}{price} ({change}%)", key=f"btn_{ticker}", use_container_width=True):
                    st.session_state.selected_stock = ticker
                    st.rerun()