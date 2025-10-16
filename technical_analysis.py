from typing import List, Tuple

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