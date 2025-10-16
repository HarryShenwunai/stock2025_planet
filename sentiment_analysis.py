import logging
from typing import List
from textblob import TextBlob
from data_structures import NewsData

logger = logging.getLogger(__name__)

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
