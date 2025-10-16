from dataclasses import dataclass
from typing import Dict, Optional, Any
import time

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