import sqlite3
import json
import logging
import numpy as np
import requests
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from data_structures import MarketData, AnalysisResult, DataCache
from technical_analysis import TechnicalAnalyzer
from sentiment_analysis import SentimentAnalyzer
from api_client import get_api_config

logger = logging.getLogger(__name__)

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
        try:
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
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}. Running without database.")
            self.conn = None
    
    async def fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch real-time market data"""
        cache_key = f"market_{symbol}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
            
        api_key = self.config.get('alpha_vantage_key')
            
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
        if self.conn is None:
            logger.warning("Database not available, skipping storage")
            return
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO analysis_history (symbol, timestamp, analysis_result, confidence) VALUES (?, ?, ?, ?)",
                (result.symbol, result.timestamp, json.dumps(asdict(result)), result.confidence)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")