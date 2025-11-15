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
    
    async def fetch_historical_prices(self, symbol: str, days: int = 100) -> List[float]:
        """Fetch real historical price data from Alpha Vantage"""
        cache_key = f"history_{symbol}_{days}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        api_key = self.config.get('alpha_vantage_key')
        
        try:
            # Use TIME_SERIES_DAILY for historical data
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=compact"
            response = requests.get(url, timeout=15)
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            
            data = response.json()
            
            if "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                # Extract closing prices, sorted from oldest to newest
                dates = sorted(time_series.keys())[-days:]
                prices = [float(time_series[date]["4. close"]) for date in dates]
                
                if len(prices) >= 14:  # Minimum for RSI calculation
                    self.cache.set(cache_key, prices, ttl=3600)  # Cache for 1 hour
                    logger.info(f"Fetched {len(prices)} historical prices for {symbol}")
                    return prices
                else:
                    raise Exception(f"Insufficient data: only {len(prices)} days available")
            else:
                # Check for API limit error
                if "Note" in data or "Information" in data:
                    logger.warning(f"API limit reached: {data.get('Note') or data.get('Information')}")
                    raise Exception("API rate limit exceeded")
                raise Exception("Invalid historical data format")
        
        except Exception as e:
            logger.warning(f"Failed to fetch real historical data for {symbol}: {e}")
            # Fallback: Generate more realistic synthetic data with random walk
            logger.info(f"Using enhanced synthetic data for {symbol}")
            return None
    
    def _generate_synthetic_prices(self, current_price: float, days: int = 100) -> List[float]:
        """Generate realistic synthetic price data using random walk with drift"""
        np.random.seed(None)  # Ensure randomness
        prices = [current_price]
        
        # Parameters for more realistic simulation
        daily_return_mean = 0.0005  # ~0.05% daily average (12% annual)
        daily_volatility = 0.015    # ~1.5% daily volatility (24% annual)
        
        for _ in range(days - 1):
            # Geometric Brownian Motion: S(t+1) = S(t) * exp(μ*dt + σ*sqrt(dt)*Z)
            random_shock = np.random.normal(0, 1)
            daily_return = daily_return_mean + daily_volatility * random_shock
            next_price = prices[-1] * (1 + daily_return)
            prices.append(max(next_price, current_price * 0.5))  # Floor at 50% of current
        
        # Reverse to oldest-to-newest order, ending near current price
        prices.reverse()
        
        # Adjust last price to match current (smooth transition)
        adjustment = current_price / prices[-1]
        prices = [p * adjustment for p in prices]
        
        return prices
    
    async def perform_comprehensive_analysis(self, symbol: str) -> AnalysisResult:
        """Perform comprehensive AI analysis"""
        logger.info(f"Starting comprehensive analysis for {symbol}")
        
        market_data = await self.fetch_market_data(symbol)
        if not market_data:
            raise ValueError(f"Could not fetch market data for {symbol}")
        
        # Try to fetch real historical prices
        historical_prices = await self.fetch_historical_prices(symbol, days=100)
        
        # Fallback to synthetic data if real data unavailable
        if historical_prices is None or len(historical_prices) < 14:
            logger.warning(f"Using synthetic price data for {symbol}")
            historical_prices = self._generate_synthetic_prices(market_data.close_price, days=100)
            data_source = "synthetic"
        else:
            data_source = "real"
            logger.info(f"Using {len(historical_prices)} days of real historical data")
        
        # Technical Analysis
        rsi = self.technical_analyzer.calculate_rsi(historical_prices)
        macd, signal, histogram = self.technical_analyzer.calculate_macd(historical_prices)
        # Advanced scientific analysis
        try:
            comp = self.technical_analyzer.comprehensive_analysis(historical_prices)
        except Exception as _:
            comp = {}
        pattern_info = comp.get("pattern_recognition") or {}
        pattern = pattern_info.get("pattern") if isinstance(pattern_info, dict) else str(pattern_info or "unknown")

        stat = comp.get("statistical_analysis") or {}
        hurst = float(((stat.get("hurst_exponent") or {}).get("value", 0.5)))
        fractal_dim = float(((stat.get("fractal_dimension") or {}).get("value", 1.5)))
        entropy = float(((stat.get("entropy") or {}).get("value", 0.0)))
        linreg = stat.get("linear_regression") or {}
        trend_r2 = float(linreg.get("r_squared", 0.0))

        risk_metrics = comp.get("risk_metrics") or {}
        sharpe = float(((risk_metrics.get("sharpe_ratio") or {}).get("value", 0.0)))
        sortino = float(((risk_metrics.get("sortino_ratio") or {}).get("value", 0.0)))
        max_dd = risk_metrics.get("maximum_drawdown") or {}
        max_drawdown_pct = float(max_dd.get("max_drawdown_pct", 0.0))

        cyc = comp.get("cyclical_analysis") or {}
        dominant = (cyc.get("dominant_periods") or [])
        cycle_period = float(dominant[0]["period_days"]) if dominant else 0.0
        
        # Risk Assessment
        volatility = self.calculate_volatility(historical_prices)
        risk_level = self.assess_risk_level(volatility, rsi, 0.1)
        
        # Generate Recommendation
        recommendation = self.generate_recommendation(market_data, rsi, macd, 0.1, volatility)

        # Compose a composite multi-factor score ([-inf, inf], higher better)
        # momentum( macd, rsi ), trend (r2), risk-adjusted (sharpe, sortino),
        # risk penalty (max drawdown, volatility), information quality (entropy)
        momentum = (1 if macd > 0 else -1 if macd < 0 else 0) + (1 - abs(rsi - 50) / 50)  # 0~2
        trend_quality = trend_r2  # 0~1
        risk_adjusted = max(sharpe, 0) * 0.5 + max(sortino, 0) * 0.5  # >=0 preferred
        risk_penalty = (max_drawdown_pct / 100.0) * 0.8 + (volatility / 100.0) * 0.2  # normalize
        info_quality = max(0.0, 3.5 - entropy) / 3.5  # 0~1，越低熵越好
        composite_score = round(1.2 * momentum + 1.0 * trend_quality + 1.0 * risk_adjusted + 0.6 * info_quality - 1.2 * risk_penalty, 3)
        
        # Calculate prices
        target_price = self.calculate_target_price(market_data, recommendation)
        stop_loss = self.calculate_stop_loss(market_data, risk_level)
        
        # Generate reasoning with data source info
        reasoning = self.generate_reasoning(market_data, rsi, macd, 0.1, pattern, recommendation)
        reasoning += f" [Data source: {data_source}, {len(historical_prices)} days]"
        
        # Adjust confidence based on data source
        base_confidence = 0.75
        if data_source == "synthetic":
            confidence = base_confidence * 0.6  # Lower confidence for synthetic data
        else:
            confidence = base_confidence
        
        result = AnalysisResult(
            symbol=symbol,
            current_price=market_data.close_price,
            trend=self.determine_trend(historical_prices),
            confidence=confidence,
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
                "volatility": volatility,
                # Extended metrics for richer selection
                "hurst": hurst,
                "fractal_dimension": fractal_dim,
                "entropy": entropy,
                "trend_r2": trend_r2,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown_pct": max_drawdown_pct,
                "cycle_period": cycle_period,
                "composite_score": composite_score,
                "data_source": data_source,  # Add data source indicator
                "history_days": len(historical_prices)
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