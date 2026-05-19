import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import streamlit as st
import feedparser  
import urllib.parse

@st.cache_data(ttl=3600) 
def get_news_sentiment(ticker):
    try:
        analyzer = SentimentIntensityAnalyzer()
        news_data = []
        total_score = 0
        
        # 1. YAHOO FINANCE
        stock = yf.Ticker(ticker)
        yf_news = stock.news 
        
        if yf_news:
            for article in yf_news:
                title = article.get('title', '')
                if not title or str(title).strip() == "": continue
                publisher = article.get('publisher', 'Yahoo Finance')
                link = article.get('link', '#')
                
                sentiment = analyzer.polarity_scores(title)
                compound_score = sentiment['compound'] 
                total_score += compound_score
                label = "🟢 Positive" if compound_score >= 0.05 else "🔴 Negative" if compound_score <= -0.05 else "🟡 Neutral"
                news_data.append({"Headline": title, "Publisher": publisher, "Sentiment": label, "Score": round(compound_score, 2), "Link": link})

        # 2. GOOGLE NEWS FALLBACK
        if not news_data:
            query = urllib.parse.quote(f"{ticker} stock news")
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:10]:
                title = entry.title
                link = entry.link
                publisher = entry.source.title if hasattr(entry, 'source') else "Google News RSS"
                
                sentiment = analyzer.polarity_scores(title)
                compound_score = sentiment['compound'] 
                total_score += compound_score
                label = "🟢 Positive" if compound_score >= 0.05 else "🔴 Negative" if compound_score <= -0.05 else "🟡 Neutral"
                news_data.append({"Headline": title, "Publisher": publisher, "Sentiment": label, "Score": round(compound_score, 2), "Link": link})

        # 3. CONSENSUS
        if not news_data: return pd.DataFrame(), 0, "Neutral"
            
        df_news = pd.DataFrame(news_data)
        avg_score = total_score / len(news_data) 
        
        if avg_score >= 0.05: overall_consensus = "Bullish (Positive)"
        elif avg_score <= -0.05: overall_consensus = "Bearish (Negative)"
        else: overall_consensus = "Neutral"
            
        return df_news, avg_score, overall_consensus
        
    except Exception as e:
        print(f"NLP Engine Error: {e}")
        return pd.DataFrame(), 0, "Neutral"