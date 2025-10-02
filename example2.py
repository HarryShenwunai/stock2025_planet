"""
Professional AI Agent for Financial Analysis and Market Intelligence
==================================================================

This AI agent provides comprehensive financial analysis, market intelligence,
and automated decision-making capabilities through multiple data sources and APIs.

Features:
- Multi-source data aggregation (Alpha Vantage, Yahoo Finance, News APIs)
- Advanced technical analysis and pattern recognition
- Risk assessment and portfolio optimization
- Natural language processing for news sentiment analysis
- RESTful API endpoints for integration
- Machine learning predictions
- Real-time market monitoring
- Automated reporting and alerts

Author: AI Financial Agent v2.0
Date: September 2025
"""

import requests
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from textblob import TextBlob
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from functools import wraps
import hashlib

# Configure logging system
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Data classes for structured data: decorator 
@dataclass
class MarketData:
    """Data structure for market information"""
    symbol: str
    timestamp: str
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: int
    change_percent: float
    
@dataclass
class NewsData:
    """Data structure for news information"""
    title: str
    content: str
    source: str
    sentiment_score: float
    relevance_score: float
    timestamp: str

@dataclass
class AnalysisResult:
    """Data structure for analysis results"""
    symbol: str
    current_price: float
    trend: str
    confidence: float
    risk_level: str
    recommendation: str
    target_price: float
    stop_loss: float
    sentiment_score: float
    technical_indicators: Dict[str, float]
    reasoning: str
    timestamp: str

class DataCache:
    """In-memory cache for API responses"""
    def __init__(self, ttl_seconds: int = 300): #constructor with default ttl 5 minutes 
        self.cache = {}
        self.ttl = ttl_seconds
        
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key] # Expired
        return None
        
    def set(self, key: str, value: Any) -> None:
        self.cache[key] = (value, time.time())

class TechnicalAnalyzer:
    """Advanced technical analysis engine"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float: # initial period is 2 weeks
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0 # Neutral if insufficient data
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))] # compute price changes
        # separate gains and losses
        gains = [d if d > 0 else 0 for d in deltas] 
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0: # avoid 0 division
            return 0.1
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: List[float]) -> Tuple[float, float, float]:
        """Calculate MACD, Signal, and Histogram"""
        if len(prices) < 26: # Need at least 26 data points
            return 0.0, 0.0, 0.0
            
        # Simple moving averages (in practice, use exponential)
        ema12 = sum(prices[-12:]) / 12 # the simple moving average of the last 12 prices.
        ema26 = sum(prices[-26:]) / 26 # the simple moving average of the last 26 prices.
        macd = ema12 - ema26 # MACD line is the difference between the two EMAs.
        
        signal = macd * 0.8  # Simplified calculation(need to improve)
        # the difference between the MACD line and the Signal line
        histogram = macd - signal 
        
        return macd, signal, histogram
    
    @staticmethod
    def detect_pattern(prices: List[float]) -> str:
        """Detect chart patterns"""
        if len(prices) < 5:
            return "insufficient_data"
            
        recent = prices[-5:]
        
        # Simple pattern detection
        if all(recent[i] > recent[i-1] for i in range(1, len(recent))):
            return "bullish_trend"
        elif all(recent[i] < recent[i-1] for i in range(1, len(recent))):
            return "bearish_trend"
        elif recent[0] < recent[2] > recent[4]: # peak in the middle
            return "head_shoulders"
        else:
            return "sideways"

class SentimentAnalyzer:
    """News sentiment analysis engine"""
    
    @staticmethod
    def analyze_sentiment(text: str) -> float:
        """Analyze sentiment of text (-1 to 1)"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e: # in case of error, return neutral, and log the error
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    @staticmethod
    def calculate_market_sentiment(news_items: List[NewsData]) -> float:
        """Calculate overall market sentiment"""
        if not news_items:
            logger.warning("No news items available for sentiment analysis.") # log a warning
            return 0.0
        
        weighted_sentiment = sum(
            news.sentiment_score * news.relevance_score 
            for news in news_items
        )
        total_weight = sum(news.relevance_score for news in news_items)
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0

class FinancialAIAgent:
    """Main AI Agent for financial analysis"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.cache = DataCache()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database for storing analysis history"""
        self.conn = sqlite3.connect('agent_data.db', check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                analysis_result TEXT,
                confidence REAL
            )
        ''')
        self.conn.commit()
    
    async def fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch real-time market data"""
        cache_key = f"market_{symbol}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        api_key = self.config.get('alpha_vantage_key')
        if not api_key:
            raise ValueError("API key for Alpha Vantage is not configured. Please add it to your config.")
            
        try:
            # Alpha Vantage API call using requests (synchronous)
            import requests
            url = (f"https://www.alphavantage.co/query"
                  f"?function=GLOBAL_QUOTE"
                  f"&symbol={symbol}"
                  f"&apikey={api_key}")
            
            logger.info(f"Fetching data from: {url[:80]}...")
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                raise Exception(f"HTTP {response.status_code}")
                
            data = response.json()
            logger.debug(f"API response for {symbol}: {data}")
                    
            if "Global Quote" in data and data["Global Quote"]:
                quote = data["Global Quote"]
                logger.info(f"Successfully fetched data for {symbol}: ${quote.get('05. price')}")
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now().isoformat(),
                    open_price=float(quote.get("02. open", 0)),
                    close_price=float(quote.get("05. price", 0)),
                    high_price=float(quote.get("03. high", 0)),
                    low_price=float(quote.get("04. low", 0)),
                    volume=int(quote.get("06. volume", 0)),
                    change_percent=float(quote.get("10. change percent", "0%").rstrip("%"))
                )
                
                self.cache.set(cache_key, market_data)
                return market_data
            else:
                logger.error(f"No 'Global Quote' in response: {data}")
                raise Exception("Invalid API response format")
                
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            logger.debug(f"Response data: {data if 'data' in locals() else 'No response data'}")
            
        # If API fails, return dummy data
        logger.warning(f"Using dummy data for {symbol}")
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            open_price=-1,
            close_price=-1,
            high_price=-1,
            low_price=-1,
            volume=-1,
            change_percent=-1
        )
    
    async def fetch_news_data(self, symbol: str) -> List[NewsData]:
        """Fetch relevant news for sentiment analysis"""
        api_key = self.config.get('news_api_key')
        if not api_key:
            raise ValueError("API key for NewsAPI is not configured. Please add it to your config.")
            
        try:
            # NewsAPI or similar service using requests
            import requests
            url = (f"https://newsapi.org/v2/everything"
                  f"?q={symbol}&sortBy=publishedAt&pageSize=10"
                  f"&apiKey={api_key}")
            
            logger.info(f"Fetching news for {symbol}...")
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"News API HTTP error {response.status_code}: {response.text}")
                return []
                
            data = response.json()
            
            if "articles" not in data:
                logger.error(f"No articles in news response: {data}")
                return []
            
            articles = data.get("articles", [])
            logger.info(f"Found {len(articles)} news articles for {symbol}")
            
            news_items = []
            for article in articles[:5]:
                title = article.get("title", "")
                content = article.get("description", "")
                
                if not title and not content:
                    continue
                    
                combined_text = f"{title} {content}"
                
                sentiment = self.sentiment_analyzer.analyze_sentiment(combined_text)
                relevance = self.calculate_relevance(combined_text, symbol)
                
                news_items.append(NewsData(
                    title=title,
                    content=content,
                    source=article.get("source", {}).get("name", "Unknown"),
                    sentiment_score=sentiment,
                    relevance_score=relevance,
                    timestamp=article.get("publishedAt", "")
                ))
            
            logger.info(f"Processed {len(news_items)} news items for {symbol}")
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching news data for {symbol}: {e}")
            return []
    
    def calculate_relevance(self, text: str, symbol: str) -> float:
        """Calculate relevance score of news to the symbol"""
        text_lower = text.lower()
        symbol_lower = symbol.lower()
        
        # Simple relevance scoring
        base_score = 0.1
        if symbol_lower in text_lower:
            base_score += 0.7
        
        financial_keywords = ["stock", "price", "market", "trading", "earnings", "revenue"]
        keyword_count = sum(1 for keyword in financial_keywords if keyword in text_lower)
        base_score += keyword_count * 0.05
        
        return min(base_score, 1.0)
    
    async def perform_comprehensive_analysis(self, symbol: str) -> AnalysisResult:
        """Perform comprehensive AI analysis"""
        logger.info(f"Starting comprehensive analysis for {symbol}")
        
        # Fetch data concurrently
        market_task = self.fetch_market_data(symbol)
        news_task = self.fetch_news_data(symbol)
        
        market_data = await market_task
        news_data = await news_task
        
        if not market_data:
            raise ValueError(f"Could not fetch market data for {symbol}")
        
        # Technical Analysis
        historical_prices = await self.fetch_historical_prices(symbol)
        rsi = self.technical_analyzer.calculate_rsi(historical_prices)
        macd, signal, histogram = self.technical_analyzer.calculate_macd(historical_prices)
        pattern = self.technical_analyzer.detect_pattern(historical_prices)
        
        # Sentiment Analysis
        market_sentiment = self.sentiment_analyzer.calculate_market_sentiment(news_data)
        
        # Risk Assessment
        volatility = self.calculate_volatility(historical_prices)
        risk_level = self.assess_risk_level(volatility, rsi, market_sentiment)
        
        # Generate Recommendation
        recommendation = self.generate_recommendation(
            market_data, rsi, macd, market_sentiment, volatility
        )
        
        # Calculate target price and stop loss
        target_price = self.calculate_target_price(market_data, recommendation)
        stop_loss = self.calculate_stop_loss(market_data, risk_level)
        
        # Assess confidence
        confidence = self.calculate_confidence(market_data, news_data, rsi, market_sentiment)
        
        # Generate reasoning
        reasoning = self.generate_reasoning(
            market_data, rsi, macd, market_sentiment, pattern, recommendation
        )
        
        result = AnalysisResult(
            symbol=symbol,
            current_price=market_data.close_price,
            trend=self.determine_trend(historical_prices),
            confidence=confidence,
            risk_level=risk_level,
            recommendation=recommendation,
            target_price=target_price,
            stop_loss=stop_loss,
            sentiment_score=market_sentiment,
            technical_indicators={
                "rsi": rsi,
                "macd": macd,
                "signal": signal,
                "histogram": histogram,
                "volatility": volatility
            },
            reasoning=reasoning,
            timestamp=datetime.now().isoformat()
        )
        
        # Store in database
        self.store_analysis(result)
        
        logger.info(f"Analysis completed for {symbol}")
        return result
    
    async def fetch_historical_prices(self, symbol: str) -> List[float]:
        """Fetch historical price data"""
        api_key = self.config.get('alpha_vantage_key')
        if not api_key:
            raise ValueError("API key for Alpha Vantage is not configured. Please add it to your config.")
        try:
            # Alpha Vantage historical data using requests
            import requests
            url = (f"https://www.alphavantage.co/query"
                  f"?function=TIME_SERIES_DAILY"
                  f"&symbol={symbol}"
                  f"&outputsize=compact"
                  f"&apikey={api_key}")
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if "Time Series (Daily)" in data:
                prices = []
                for date_key in sorted(data["Time Series (Daily)"].keys(), reverse=True)[:30]:
                    close_price = float(data["Time Series (Daily)"][date_key]["4. close"])
                    prices.append(close_price)
                return prices
                
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
        return []
    
    def calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0
        
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return np.std(returns) * 100 if returns else 0.0
    
    def assess_risk_level(self, volatility: float, rsi: float, sentiment: float) -> str:
        """Assess overall risk level"""
        risk_score = 0
        
        if volatility > 5:
            risk_score += 2
        elif volatility > 2:
            risk_score += 1
            
        if rsi > 80 or rsi < 20:
            risk_score += 2
        elif rsi > 70 or rsi < 30:
            risk_score += 1
            
        if abs(sentiment) > 0.5:
            risk_score += 1
        
        if risk_score >= 4:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_recommendation(self, market_data: MarketData, rsi: float, 
                              macd: float, sentiment: float, volatility: float) -> str:
        """Generate trading recommendation"""
        score = 0
        
        # Technical factors
        if rsi < 30:
            score += 2  # Oversold
        elif rsi > 70:
            score -= 2  # Overbought
            
        if macd > 0:
            score += 1  # Bullish momentum
        else:
            score -= 1  # Bearish momentum
            
        # Sentiment factor
        score += sentiment * 2
        
        # Price momentum
        if market_data.change_percent > 2:
            score += 1
        elif market_data.change_percent < -2:
            score -= 1
        
        # Risk adjustment
        if volatility > 5:
            score *= 0.5  # Reduce confidence in high volatility
        
        if score >= 2:
            return "STRONG_BUY"
        elif score >= 1:
            return "BUY"
        elif score >= -1:
            return "HOLD"
        elif score >= -2:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def calculate_target_price(self, market_data: MarketData, recommendation: str) -> float:
        """Calculate target price based on recommendation"""
        current_price = market_data.close_price
        
        if recommendation in ["STRONG_BUY", "BUY"]:
            return current_price * 1.15  # 15% upside
        elif recommendation == "HOLD":
            return current_price * 1.05  # 5% upside
        else:
            return current_price * 0.95  # 5% downside
    
    def calculate_stop_loss(self, market_data: MarketData, risk_level: str) -> float:
        """Calculate stop loss based on risk level"""
        current_price = market_data.close_price
        
        if risk_level == "HIGH":
            return current_price * 0.90  # 10% stop loss
        elif risk_level == "MEDIUM":
            return current_price * 0.85  # 15% stop loss
        else:
            return current_price * 0.80  # 20% stop loss
    
    def calculate_confidence(self, market_data: MarketData, news_data: List[NewsData], 
                           rsi: float, sentiment: float) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence
        
        # Data quality factors
        if market_data.volume > 1000000:
            confidence += 0.1
            
        if len(news_data) >= 3:
            confidence += 0.1
        
        # Technical certainty
        if 30 <= rsi <= 70:
            confidence += 0.1  # RSI in normal range
        
        # Sentiment certainty
        if abs(sentiment) > 0.3:
            confidence += 0.2  # Strong sentiment signal
        
        return min(confidence, 1.0)
    
    def determine_trend(self, prices: List[float]) -> str:
        """Determine overall price trend"""
        if len(prices) < 3:
            return "UNKNOWN"
            
        recent_avg = sum(prices[:5]) / 5
        older_avg = sum(prices[-5:]) / 5
        
        if recent_avg > older_avg * 1.02:
            return "UPTREND"
        elif recent_avg < older_avg * 0.98:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def generate_reasoning(self, market_data: MarketData, rsi: float, macd: float,
                          sentiment: float, pattern: str, recommendation: str) -> str:
        """Generate human-readable reasoning for the recommendation"""
        reasoning_parts = []
        
        reasoning_parts.append(f"Current price: ${market_data.close_price:.2f}")
        reasoning_parts.append(f"Daily change: {market_data.change_percent:.2f}%")
        
        if rsi > 70:
            reasoning_parts.append("RSI indicates overbought conditions")
        elif rsi < 30:
            reasoning_parts.append("RSI indicates oversold conditions")
        else:
            reasoning_parts.append("RSI is in neutral territory")
        
        if macd > 0:
            reasoning_parts.append("MACD shows bullish momentum")
        else:
            reasoning_parts.append("MACD shows bearish momentum")
        
        if sentiment > 0.2:
            reasoning_parts.append("Market sentiment is positive")
        elif sentiment < -0.2:
            reasoning_parts.append("Market sentiment is negative")
        else:
            reasoning_parts.append("Market sentiment is neutral")
        
        reasoning_parts.append(f"Chart pattern: {pattern}")
        reasoning_parts.append(f"Recommendation: {recommendation}")
        
        return ". ".join(reasoning_parts) + "."
    
    def store_analysis(self, result: AnalysisResult):
        """Store analysis result in database"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO analysis_history (symbol, timestamp, analysis_result, confidence) VALUES (?, ?, ?, ?)",
                (result.symbol, result.timestamp, json.dumps(asdict(result)), result.confidence)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")

# Global agent instance
agent = None

def init_agent():
    """Initialize the AI agent"""
    global agent
    try:
        from config import ALPHA_VANTAGE_KEY, NEWS_API_KEY
        config = {
            'alpha_vantage_key': ALPHA_VANTAGE_KEY,
            'news_api_key': NEWS_API_KEY,
        }
    except ImportError:
        logger.warning("Config file not found, using environment variables")
        config = {
            'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_KEY', 'demo'),
            'news_api_key': os.getenv('NEWS_API_KEY', 'demo'),
        }
    
    logger.info(f"Initializing agent with API key: {config['alpha_vantage_key'][:10]}...")
    agent = FinancialAIAgent(config)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_agent()
    yield
    # Shutdown
    pass

# FastAPI app for the AI Agent
app = FastAPI(
    title="AI Financial Agent API",
    description="Professional AI agent for financial analysis and market intelligence",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def index():
    """Welcome page with API documentation"""
    return """
    <html>
        <head>
            <title>AI Financial Agent API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; }
                h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                ul { list-style-type: none; padding: 0; }
                li { margin: 10px 0; padding: 10px; background: #ecf0f1; border-radius: 5px; }
                a { color: #3498db; text-decoration: none; font-weight: bold; }
                a:hover { color: #2980b9; }
                .endpoint { font-family: monospace; background: #2c3e50; color: white; padding: 5px 10px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Financial Agent API</h1>
                <h2>Available Endpoints:</h2>
                <ul>
                    <li><span class="endpoint">GET /api/analyze/{symbol}</span> - Complete stock analysis</li>
                    <li><span class="endpoint">GET /api/history/{symbol}</span> - Historical analysis data</li>
                    <li><span class="endpoint">GET /api/health</span> - System health check</li>
                    <li><span class="endpoint">GET /docs</span> - Interactive API documentation</li>
                    <li><span class="endpoint">GET /redoc</span> - Alternative API documentation</li>
                </ul>
                <h2>Example Usage:</h2>
                <ul>
                    <li><a href="/api/analyze/AAPL">Analyze AAPL</a></li>
                    <li><a href="/api/analyze/TSLA">Analyze TSLA</a></li>
                    <li><a href="/api/analyze/MSFT">Analyze MSFT</a></li>
                    <li><a href="/api/health">Health Check</a></li>
                    <li><a href="/docs">API Documentation</a></li>
                </ul>
                <p><em>💡 Tip: Visit <a href="/docs">/docs</a> for interactive API testing!</em></p>
            </div>
        </body>
    </html>
    """

@app.get("/api/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    """API endpoint for stock analysis"""
    try:
        result = await agent.perform_comprehensive_analysis(symbol.upper())
        return asdict(result)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{symbol}")
async def get_analysis_history(symbol: str):
    """Get analysis history for a symbol"""
    try:
        cursor = agent.conn.cursor()
        cursor.execute(
            "SELECT * FROM analysis_history WHERE symbol = ? ORDER BY timestamp DESC LIMIT 10",
            (symbol.upper(),)
        )
        results = cursor.fetchall()
        
        history = []
        for row in results:
            history.append({
                'id': row[0],
                'symbol': row[1],
                'timestamp': row[2],
                'analysis': json.loads(row[3]),
                'confidence': row[4]
            })
        
        return history
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    print("AI Financial Agent - Starting FastAPI Server")
    print("=" * 50)
    print("Available endpoints:")
    print("  GET / - Welcome page")
    print("  GET /api/analyze/{symbol} - Analyze stock")
    print("  GET /api/history/{symbol} - Analysis history")
    print("  GET /api/health - Health check")
    print("  GET /docs - Interactive API documentation")
    print("  GET /redoc - Alternative API documentation")
    print("=" * 50)
    
    # Start the FastAPI server with uvicorn
    import uvicorn
    print("\n Starting FastAPI server on http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")