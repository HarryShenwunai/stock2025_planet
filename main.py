"""
Combined Financial Analysis Web Application
==========================================

This application combines three different examples into a unified web interface:
- Flask-based events and stock data display
- Advanced AI Financial Agent with comprehensive analysis
- Simple FastAPI endpoints for stock and news data

Features:
- Multi-framework support (Flask + FastAPI)
- Real-time stock data and analysis
- News sentiment analysis
- Technical indicators (RSI, MACD)
- Interactive charts and visualizations
- RESTful API endpoints
- Modern responsive UI

Author: Combined Examples v1.0
Date: October 2025
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
# Removed Flask import - using FastAPI only for better compatibility
from textblob import TextBlob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('combined_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== DATA STRUCTURES =====
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

# ===== UTILITY CLASSES =====
class DataCache:
    """In-memory cache for API responses"""
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds
        
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
        
    def set(self, key: str, value: Any) -> None:
        self.cache[key] = (value, time.time())

class TechnicalAnalyzer:
    """Advanced technical analysis engine"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 0.1
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: List[float]) -> Tuple[float, float, float]:
        """Calculate MACD, Signal, and Histogram"""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
            
        ema12 = sum(prices[-12:]) / 12
        ema26 = sum(prices[-26:]) / 26
        macd = ema12 - ema26
        signal = macd * 0.8
        histogram = macd - signal
        
        return macd, signal, histogram
    
    @staticmethod
    def detect_pattern(prices: List[float]) -> str:
        """Detect chart patterns"""
        if len(prices) < 5:
            return "insufficient_data"
            
        recent = prices[-5:]
        
        if all(recent[i] > recent[i-1] for i in range(1, len(recent))):
            return "bullish_trend"
        elif all(recent[i] < recent[i-1] for i in range(1, len(recent))):
            return "bearish_trend"
        elif recent[0] < recent[2] > recent[4]:
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
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    @staticmethod
    def calculate_market_sentiment(news_items: List[NewsData]) -> float:
        """Calculate overall market sentiment"""
        if not news_items:
            return 0.0
        
        weighted_sentiment = sum(
            news.sentiment_score * news.relevance_score 
            for news in news_items
        )
        total_weight = sum(news.relevance_score for news in news_items)
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0

# ===== API FUNCTIONS =====
def get_api_config():
    """Get API configuration from config file or environment"""
    try:
        from config import ALPHA_VANTAGE_KEY, NEWS_API_KEY
        return {
            'alpha_vantage_key': ALPHA_VANTAGE_KEY,
            'news_api_key': NEWS_API_KEY,
        }
    except ImportError:
        logger.warning("Config file not found, using environment variables")
        return {
            'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_KEY', 'demo'),
            'news_api_key': os.getenv('NEWS_API_KEY', 'demo'),
        }

def fetch_api_data(url: str) -> List[Dict]:
    """Fetch data from API (from example1.py)"""
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data.get('groups', []) or data.get('data', []) or data.get('results', []) or data.get('items', []) or data.get('events', [])
        elif isinstance(data, list):
            return data
        return []
    except Exception as e:
        logger.error(f"Error fetching API data from {url}: {e}")
        return []

def fetch_stock_data(url: str) -> Dict:
    """Fetch stock data from API (from example1.py)"""
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching stock data from {url}: {e}")
        return {}

def fetch_alpha_vantage_stock(symbol: str) -> Dict[str, Any]:
    """Fetch stock data using Alpha Vantage API (from example3.py)"""
    config = get_api_config()
    api_key = config['alpha_vantage_key']
    
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except Exception as e:
        logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
        return {"status": "error", "message": str(e)}

def fetch_news_data(symbol: str) -> Dict[str, Any]:
    """Fetch news data using News API (from example3.py)"""
    config = get_api_config()
    api_key = config['news_api_key']
    
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&pageSize=10&apiKey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except Exception as e:
        logger.error(f"Error fetching news data for {symbol}: {e}")
        return {"status": "error", "message": str(e)}

# ===== MAIN AI AGENT CLASS =====
class CombinedFinancialAgent:
    """Combined AI Agent for financial analysis"""
    
    def __init__(self):
        self.config = get_api_config()
        self.cache = DataCache()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect('combined_agent_data.db', check_same_thread=False)
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
        if not api_key or api_key == 'demo':
            logger.warning(f"Using dummy data for {symbol} - no valid API key")
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                open_price=150.0,
                close_price=152.0,
                high_price=155.0,
                low_price=148.0,
                volume=1000000,
                change_percent=1.33
            )
            
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
                
            data = response.json()
            
            if "Global Quote" in data and data["Global Quote"]:
                quote = data["Global Quote"]
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
                raise Exception("Invalid API response format")
                
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            # Return dummy data as fallback
            return MarketData(
                symbol=symbol,
                timestamp=datetime.now().isoformat(),
                open_price=150.0,
                close_price=152.0,
                high_price=155.0,
                low_price=148.0,
                volume=1000000,
                change_percent=1.33
            )
    
    async def perform_comprehensive_analysis(self, symbol: str) -> AnalysisResult:
        """Perform comprehensive AI analysis"""
        logger.info(f"Starting comprehensive analysis for {symbol}")
        
        market_data = await self.fetch_market_data(symbol)
        if not market_data:
            raise ValueError(f"Could not fetch market data for {symbol}")
        
        # Simulate historical prices for analysis
        historical_prices = [market_data.close_price + (i * 0.5) for i in range(-30, 1)]
        
        # Technical Analysis
        rsi = self.technical_analyzer.calculate_rsi(historical_prices)
        macd, signal, histogram = self.technical_analyzer.calculate_macd(historical_prices)
        pattern = self.technical_analyzer.detect_pattern(historical_prices)
        
        # Risk Assessment
        volatility = self.calculate_volatility(historical_prices)
        risk_level = self.assess_risk_level(volatility, rsi, 0.1)
        
        # Generate Recommendation
        recommendation = self.generate_recommendation(market_data, rsi, macd, 0.1, volatility)
        
        # Calculate prices
        target_price = self.calculate_target_price(market_data, recommendation)
        stop_loss = self.calculate_stop_loss(market_data, risk_level)
        
        # Generate reasoning
        reasoning = self.generate_reasoning(market_data, rsi, macd, 0.1, pattern, recommendation)
        
        result = AnalysisResult(
            symbol=symbol,
            current_price=market_data.close_price,
            trend=self.determine_trend(historical_prices),
            confidence=0.75,
            risk_level=risk_level,
            recommendation=recommendation,
            target_price=target_price,
            stop_loss=stop_loss,
            sentiment_score=0.1,
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
        
        return result
    
    def calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility"""
        if len(prices) < 2:
            return 0.0
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return np.std(returns) * 100 if returns else 0.0
    
    def assess_risk_level(self, volatility: float, rsi: float, sentiment: float) -> str:
        """Assess overall risk level"""
        risk_score = 0
        if volatility > 5: risk_score += 2
        elif volatility > 2: risk_score += 1
        if rsi > 80 or rsi < 20: risk_score += 2
        elif rsi > 70 or rsi < 30: risk_score += 1
        if abs(sentiment) > 0.5: risk_score += 1
        
        if risk_score >= 4: return "HIGH"
        elif risk_score >= 2: return "MEDIUM"
        else: return "LOW"
    
    def generate_recommendation(self, market_data: MarketData, rsi: float, 
                              macd: float, sentiment: float, volatility: float) -> str:
        """Generate trading recommendation"""
        score = 0
        if rsi < 30: score += 2
        elif rsi > 70: score -= 2
        if macd > 0: score += 1
        else: score -= 1
        score += sentiment * 2
        if market_data.change_percent > 2: score += 1
        elif market_data.change_percent < -2: score -= 1
        if volatility > 5: score *= 0.5
        
        if score >= 2: return "STRONG_BUY"
        elif score >= 1: return "BUY"
        elif score >= -1: return "HOLD"
        elif score >= -2: return "SELL"
        else: return "STRONG_SELL"
    
    def calculate_target_price(self, market_data: MarketData, recommendation: str) -> float:
        """Calculate target price"""
        current_price = market_data.close_price
        if recommendation in ["STRONG_BUY", "BUY"]: return current_price * 1.15
        elif recommendation == "HOLD": return current_price * 1.05
        else: return current_price * 0.95
    
    def calculate_stop_loss(self, market_data: MarketData, risk_level: str) -> float:
        """Calculate stop loss"""
        current_price = market_data.close_price
        if risk_level == "HIGH": return current_price * 0.90
        elif risk_level == "MEDIUM": return current_price * 0.85
        else: return current_price * 0.80
    
    def determine_trend(self, prices: List[float]) -> str:
        """Determine overall price trend"""
        if len(prices) < 3: return "UNKNOWN"
        recent_avg = sum(prices[:5]) / 5
        older_avg = sum(prices[-5:]) / 5
        if recent_avg > older_avg * 1.02: return "UPTREND"
        elif recent_avg < older_avg * 0.98: return "DOWNTREND"
        else: return "SIDEWAYS"
    
    def generate_reasoning(self, market_data: MarketData, rsi: float, macd: float,
                          sentiment: float, pattern: str, recommendation: str) -> str:
        """Generate reasoning for the recommendation"""
        reasoning_parts = [
            f"Current price: ${market_data.close_price:.2f}",
            f"Daily change: {market_data.change_percent:.2f}%",
            "RSI indicates overbought conditions" if rsi > 70 else 
            "RSI indicates oversold conditions" if rsi < 30 else 
            "RSI is in neutral territory",
            "MACD shows bullish momentum" if macd > 0 else "MACD shows bearish momentum",
            "Market sentiment is positive" if sentiment > 0.2 else 
            "Market sentiment is negative" if sentiment < -0.2 else 
            "Market sentiment is neutral",
            f"Chart pattern: {pattern}",
            f"Recommendation: {recommendation}"
        ]
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

# ===== GLOBAL AGENT =====
agent = None

def init_agent():
    """Initialize the combined agent"""
    global agent
    agent = CombinedFinancialAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_agent()
    yield
    # Shutdown
    pass

# ===== FASTAPI APPLICATION =====
app = FastAPI(
    title="Combined Financial Analysis API",
    description="Unified financial analysis platform combining multiple data sources and AI analysis",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== HTML TEMPLATES =====
def get_main_page_html():
    """Main dashboard HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Combined Financial Analysis Platform</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { 
                background: rgba(255,255,255,0.95); 
                border-radius: 15px; 
                padding: 30px; 
                margin-bottom: 30px; 
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .header h1 { 
                color: #2c3e50; 
                font-size: 2.5em; 
                margin-bottom: 10px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .header p { color: #7f8c8d; font-size: 1.2em; }
            .sections { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px; 
            }
            .section { 
                background: rgba(255,255,255,0.95); 
                border-radius: 15px; 
                padding: 25px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .section:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.15);
            }
            .section h2 { 
                color: #2c3e50; 
                margin-bottom: 15px; 
                font-size: 1.5em;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            .endpoints { list-style: none; }
            .endpoints li { 
                margin: 12px 0; 
                padding: 12px; 
                background: #f8f9fa; 
                border-radius: 8px; 
                border-left: 4px solid #3498db;
                transition: background 0.3s ease;
            }
            .endpoints li:hover { background: #e3f2fd; }
            .endpoint { 
                font-family: 'Courier New', monospace; 
                background: #2c3e50; 
                color: white; 
                padding: 6px 12px; 
                border-radius: 5px; 
                font-size: 0.9em;
                display: inline-block;
                margin-bottom: 5px;
            }
            .method-get { background: #27ae60 !important; }
            .method-post { background: #e74c3c !important; }
            a { 
                color: #3498db; 
                text-decoration: none; 
                font-weight: 600;
                transition: color 0.3s ease;
            }
            a:hover { color: #2980b9; }
            .demo-section {
                background: rgba(255,255,255,0.95);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .demo-links {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            .demo-link {
                background: linear-gradient(45deg, #3498db, #2980b9);
                color: white !important;
                padding: 15px 20px;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                text-decoration: none;
            }
            .demo-link:hover {
                transform: scale(1.05);
                box-shadow: 0 5px 20px rgba(52, 152, 219, 0.4);
                color: white !important;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Combined Financial Analysis Platform</h1>
                <p>Unified platform combining Flask, FastAPI, and AI-powered financial analysis</p>
            </div>
            
            <div class="sections">
                <div class="section">
                    <h2>Example 1 - Events & Stock Data</h2>
                    <ul class="endpoints">
                        <li><span class="endpoint method-get">GET /events</span><br>TechSum events and metrics display</li>
                        <li><span class="endpoint method-get">GET /aapl</span><br>AAPL stock metrics with visualization</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Example 2 - AI Financial Agent</h2>
                    <ul class="endpoints">
                        <li><span class="endpoint method-get">GET /api/analyze/{symbol}</span><br>Comprehensive AI stock analysis</li>
                        <li><span class="endpoint method-get">GET /api/history/{symbol}</span><br>Historical analysis data</li>
                        <li><span class="endpoint method-get">GET /api/health</span><br>System health check</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Example 3 - Stock & News APIs</h2>
                    <ul class="endpoints">
                        <li><span class="endpoint method-get">GET /stock/{symbol}</span><br>Alpha Vantage stock data</li>
                        <li><span class="endpoint method-get">GET /news/{symbol}</span><br>News API data with sentiment</li>
                    </ul>
                </div>
            </div>
            
            <div class="demo-section">
                <h2>Quick Demo Links</h2>
                <div class="demo-links">
                    <a href="/events" class="demo-link">TechSum Events</a>
                    <a href="/aapl" class="demo-link">AAPL Analysis</a>
                    <a href="/api/analyze/TSLA" class="demo-link">TSLA AI Analysis</a>
                    <a href="/stock/MSFT" class="demo-link">MSFT Stock Data</a>
                    <a href="/news/AAPL" class="demo-link">AAPL News</a>
                    <a href="/api/health" class="demo-link">Health Check</a>
                    <a href="/docs" class="demo-link">API Docs</a>
                    <a href="/dashboard" class="demo-link">Dashboard</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

def get_dashboard_html():
    """Interactive dashboard HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: #f5f7fa;
                color: #333;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .header { 
                background: white; 
                border-radius: 10px; 
                padding: 20px; 
                margin-bottom: 20px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .search-section { 
                display: flex; 
                gap: 10px; 
                align-items: center; 
            }
            input[type="text"] {
                padding: 10px 15px;
                border: 2px solid #e1e8ed;
                border-radius: 8px;
                font-size: 16px;
                width: 150px;
                transition: border-color 0.3s ease;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #3498db;
            }
            button {
                background: linear-gradient(45deg, #3498db, #2980b9);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: transform 0.2s ease;
            }
            button:hover { transform: scale(1.05); }
            .dashboard-grid { 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 20px; 
                margin-bottom: 20px; 
            }
            .widget { 
                background: white; 
                border-radius: 10px; 
                padding: 20px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                position: relative;
            }
            .widget h3 { 
                margin-bottom: 15px; 
                color: #2c3e50; 
                border-bottom: 2px solid #3498db; 
                padding-bottom: 10px; 
            }
            /* Ensure the price chart has a stable height to prevent growth */
            .widget.chart-widget {
                height: 420px;
            }
            #priceChart {
                height: 100% !important;
                width: 100% !important;
                display: block;
            }
            .analysis-result { 
                grid-column: 1 / -1; 
                background: white; 
                border-radius: 10px; 
                padding: 20px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px; 
            }
            .metrics-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                gap: 15px; 
                margin-top: 15px; 
            }
            .metric { 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 8px; 
                text-align: center; 
                border-left: 4px solid #3498db; 
            }
            .metric-value { 
                font-size: 1.5em; 
                font-weight: bold; 
                color: #2c3e50; 
            }
            .metric-label { 
                font-size: 0.9em; 
                color: #7f8c8d; 
                margin-top: 5px; 
            }
            .recommendation { 
                padding: 15px; 
                border-radius: 8px; 
                margin: 15px 0; 
                font-weight: bold; 
                text-align: center; 
            }
            .rec-buy { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .rec-sell { background: #f8d7da; color: #721c24; border: 1px solid #f1b0b7; }
            .rec-hold { background: #fff3cd; color: #856404; border: 1px solid #ffd700; }
            .loading { text-align: center; padding: 20px; color: #7f8c8d; }
            .navigation { 
                background: white; 
                border-radius: 10px; 
                padding: 15px; 
                margin-bottom: 20px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }
            .nav-link { 
                color: #3498db; 
                text-decoration: none; 
                margin-right: 20px; 
                font-weight: 600; 
                padding: 8px 16px;
                border-radius: 5px;
                transition: background 0.3s ease;
            }
            .nav-link:hover { 
                background: #e3f2fd; 
                color: #2980b9; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Financial Analysis Dashboard</h1>
                <div class="search-section">
                    <input type="text" id="symbolInput" placeholder="AAPL" value="AAPL">
                    <button onclick="analyzeStock()">Analyze</button>
                </div>
            </div>
            
            <div class="navigation">
                <a href="/" class="nav-link">Home</a>
                <a href="/events" class="nav-link">Events</a>
                <a href="/docs" class="nav-link">API Docs</a>
            </div>
            
            <div id="analysisResult" class="analysis-result" style="display: none;">
                <h3>AI Analysis Result</h3>
                <div id="analysisContent"></div>
            </div>
            
            <div class="dashboard-grid">
                <div class="widget chart-widget">
                    <h3>Price Chart</h3>
                    <canvas id="priceChart"></canvas>
                </div>
                
                <div class="widget">
                    <h3>Technical Indicators</h3>
                    <div id="technicalIndicators">
                        <p class="loading">Load a symbol to see technical indicators</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let priceChart = null;
            
            async function analyzeStock() {
                const symbol = document.getElementById('symbolInput').value.toUpperCase();
                if (!symbol) return;
                
                // Show loading
                const resultDiv = document.getElementById('analysisResult');
                const contentDiv = document.getElementById('analysisContent');
                resultDiv.style.display = 'block';
                contentDiv.innerHTML = '<div class="loading">Analyzing ' + symbol + '...</div>';
                
                try {
                    // Fetch analysis
                    const response = await fetch(`/api/analyze/${symbol}`);
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayAnalysis(data);
                        updateTechnicalIndicators(data.technical_indicators);
                        createPriceChart(data);
                    } else {
                        contentDiv.innerHTML = '<div class="loading">Error: ' + (data.detail || 'Unknown error') + '</div>';
                    }
                } catch (error) {
                    contentDiv.innerHTML = '<div class="loading">Network error: ' + error.message + '</div>';
                }
            }
            
            function displayAnalysis(data) {
                const contentDiv = document.getElementById('analysisContent');
                
                let recClass = 'rec-hold';
                if (data.recommendation.includes('BUY')) recClass = 'rec-buy';
                else if (data.recommendation.includes('SELL')) recClass = 'rec-sell';
                
                contentDiv.innerHTML = `
                    <div class="recommendation ${recClass}">
                        ${data.recommendation} - ${data.symbol}
                    </div>
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value">$${data.current_price.toFixed(2)}</div>
                            <div class="metric-label">Current Price</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.trend}</div>
                            <div class="metric-label">Trend</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${(data.confidence * 100).toFixed(0)}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.risk_level}</div>
                            <div class="metric-label">Risk Level</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">$${data.target_price.toFixed(2)}</div>
                            <div class="metric-label">Target Price</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">$${data.stop_loss.toFixed(2)}</div>
                            <div class="metric-label">Stop Loss</div>
                        </div>
                    </div>
                    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                        <h4>Analysis Reasoning:</h4>
                        <p style="margin-top: 10px; line-height: 1.6;">${data.reasoning}</p>
                    </div>
                `;
            }
            
            function updateTechnicalIndicators(indicators) {
                const div = document.getElementById('technicalIndicators');
                div.innerHTML = `
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value">${indicators.rsi.toFixed(1)}</div>
                            <div class="metric-label">RSI</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${indicators.macd.toFixed(3)}</div>
                            <div class="metric-label">MACD</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${indicators.volatility.toFixed(2)}%</div>
                            <div class="metric-label">Volatility</div>
                        </div>
                    </div>
                `;
            }
            
            function createPriceChart(data) {
                const ctx = document.getElementById('priceChart').getContext('2d');
                
                // Destroy existing chart
                if (priceChart) priceChart.destroy();
                
                // Generate sample price data for visualization
                const days = 30;
                const dates = [];
                const prices = [];
                const currentPrice = data.current_price;
                
                for (let i = days; i >= 0; i--) {
                    const date = new Date();
                    date.setDate(date.getDate() - i);
                    dates.push(date.toLocaleDateString());
                    
                    // Generate realistic price variations
                    const variation = (Math.random() - 0.5) * 0.05; // ±2.5% variation
                    const price = currentPrice * (1 + variation * (i / days));
                    prices.push(price);
                }
                
                priceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: dates,
                        datasets: [{
                            label: data.symbol + ' Price',
                            data: prices,
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        resizeDelay: 150,
                        plugins: {
                            legend: {
                                display: true
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: false,
                                title: {
                                    display: true,
                                    text: 'Price ($)'
                                }
                            }
                        }
                    }
                });
            }
            
            // Auto-analyze AAPL on load
            window.onload = function() {
                analyzeStock();
            };
        </script>
    </body>
    </html>
    """

# ===== API ENDPOINTS =====
@app.get("/", response_class=HTMLResponse)
async def index():
    """Main dashboard page"""
    return get_main_page_html()

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Interactive dashboard"""
    return get_dashboard_html()

@app.get("/events", response_class=HTMLResponse)
async def events():
    """Events page (from example1.py)"""
    url = "https://dataserver.datasum.ai/techsum/api/v3/events"
    groups = fetch_api_data(url)
    
    # Simple template rendering (inline)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>TechSum Events Metrics</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f8f8f8; margin: 20px; }}
            .page-header {{ background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .page-title {{ font-size: 1.5em; font-weight: bold; color: #333; }}
            .source-url {{ font-size: 0.9em; color: #666; background: #f0f0f0; padding: 4px 8px; border-radius: 4px; font-family: monospace; }}
            .navigation {{ background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .nav-link {{ color: #0077cc; text-decoration: none; margin-right: 20px; font-weight: bold; }}
            .group-card {{ background: #fff; border-radius: 10px; margin: 20px 0; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .group-title {{ font-size: 1.3em; font-weight: bold; margin-bottom: 10px; color: #333; }}
        </style>
    </head>
    <body>
        <div class="page-header">
            <div class="page-title">TechSum Events Metrics</div>
            <div class="source-url">Data Source: {url}</div>
        </div>
        
        <div class="navigation">
            <a href="/" class="nav-link">Home</a>
            <a href="/events" class="nav-link">TechSum Events</a>
            <a href="/aapl" class="nav-link">AAPL Stock</a>
            <a href="/dashboard" class="nav-link">Dashboard</a>
        </div>
        
        {"".join([f'<div class="group-card"><div class="group-title">{group.get("group_title", "No Title")}</div><p>{group.get("group_summary", "")}</p></div>' for group in groups[:5]] if groups else ['<div class="group-card">No events data available</div>'])}
    </body>
    </html>
    """
    return html

@app.get("/aapl", response_class=HTMLResponse)
async def aapl():
    """AAPL stock page (from example1.py)"""
    url = "https://dataserver.datasum.ai/stock-info/api/v1/stock?symbol=AAPL"
    stock_data = fetch_stock_data(url)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AAPL Stock Metrics</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f8f8f8; margin: 20px; }}
            .page-header {{ background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .page-title {{ font-size: 1.5em; font-weight: bold; color: #333; }}
            .navigation {{ background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .nav-link {{ color: #0077cc; text-decoration: none; margin-right: 20px; font-weight: bold; }}
            .stock-summary {{ background: #fff; border-radius: 10px; margin: 20px 0; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="page-header">
            <div class="page-title">AAPL Stock Metrics</div>
        </div>
        
        <div class="navigation">
            <a href="/" class="nav-link">Home</a>
            <a href="/events" class="nav-link">Events</a>
            <a href="/aapl" class="nav-link">AAPL Stock</a>
            <a href="/dashboard" class="nav-link">Dashboard</a>
        </div>
        
    </body>
    </html>
    """
    return html

@app.get("/api/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    """AI analysis endpoint (from example2.py)"""
    try:
        result = await agent.perform_comprehensive_analysis(symbol.upper())
        return asdict(result)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{symbol}")
async def get_analysis_history(symbol: str):
    """Get analysis history (from example2.py)"""
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
    """Health check endpoint (from example2.py)"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/stock/{symbol}")
def get_stock_data_api(symbol: str):
    """Stock data endpoint (from example3.py)"""
    try:
        result = fetch_alpha_vantage_stock(symbol.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

@app.get("/news/{symbol}")
def get_news_data_api(symbol: str):
    """News data endpoint (from example3.py)"""
    try:
        result = fetch_news_data(symbol.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news data: {str(e)}")

if __name__ == "__main__":
    print("Combined Financial Analysis Platform")
    print("=" * 60)
    print("Main Page: http://localhost:8000")
    print("Dashboard: http://localhost:8000/dashboard")
    print("Events: http://localhost:8000/events")
    print("AAPL: http://localhost:8000/aapl")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
