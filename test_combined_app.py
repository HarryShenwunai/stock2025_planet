"""
Comprehensive Test Suite for Combined Financial Analysis Application
================================================================

This test suite covers all major components of the combined_app.py:
- Data structures and classes
- API endpoints
- Technical analysis functions
- AI agent functionality
- Database operations
- Error handling

Author: Test Suite v1.0
Date: December 2024
"""

import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import requests
from fastapi.testclient import TestClient

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the application
from combined_app import (
    MarketData, NewsData, AnalysisResult,
    DataCache, TechnicalAnalyzer, SentimentAnalyzer,
    CombinedFinancialAgent, app, init_agent
)

class TestDataStructures(unittest.TestCase):
    """Test data structure classes"""
    
    def test_market_data_creation(self):
        """Test MarketData dataclass creation"""
        market_data = MarketData(
            symbol="AAPL",
            timestamp="2024-01-01T00:00:00",
            open_price=150.0,
            close_price=152.0,
            high_price=155.0,
            low_price=148.0,
            volume=1000000,
            change_percent=1.33
        )
        
        self.assertEqual(market_data.symbol, "AAPL")
        self.assertEqual(market_data.close_price, 152.0)
        self.assertEqual(market_data.change_percent, 1.33)
    
    def test_news_data_creation(self):
        """Test NewsData dataclass creation"""
        news_data = NewsData(
            title="Test News",
            content="Test content",
            source="Test Source",
            sentiment_score=0.5,
            relevance_score=0.8,
            timestamp="2024-01-01T00:00:00"
        )
        
        self.assertEqual(news_data.title, "Test News")
        self.assertEqual(news_data.sentiment_score, 0.5)
        self.assertEqual(news_data.relevance_score, 0.8)
    
    def test_analysis_result_creation(self):
        """Test AnalysisResult dataclass creation"""
        analysis_result = AnalysisResult(
            symbol="AAPL",
            current_price=152.0,
            trend="UPTREND",
            confidence=0.75,
            risk_level="MEDIUM",
            recommendation="BUY",
            target_price=175.0,
            stop_loss=130.0,
            sentiment_score=0.2,
            technical_indicators={"rsi": 65.0, "macd": 0.5},
            reasoning="Test reasoning",
            timestamp="2024-01-01T00:00:00"
        )
        
        self.assertEqual(analysis_result.symbol, "AAPL")
        self.assertEqual(analysis_result.confidence, 0.75)
        self.assertEqual(analysis_result.risk_level, "MEDIUM")

class TestDataCache(unittest.TestCase):
    """Test DataCache functionality"""
    
    def setUp(self):
        self.cache = DataCache(ttl_seconds=1)
    
    def test_cache_set_and_get(self):
        """Test basic cache operations"""
        self.cache.set("test_key", "test_value")
        self.assertEqual(self.cache.get("test_key"), "test_value")
    
    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        self.cache.set("test_key", "test_value")
        self.assertEqual(self.cache.get("test_key"), "test_value")
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        self.assertIsNone(self.cache.get("test_key"))
    
    def test_cache_miss(self):
        """Test cache miss for non-existent key"""
        self.assertIsNone(self.cache.get("non_existent_key"))

class TestTechnicalAnalyzer(unittest.TestCase):
    """Test TechnicalAnalyzer class"""
    
    def setUp(self):
        self.analyzer = TechnicalAnalyzer()
    
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        # Test with sufficient data
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
        rsi = self.analyzer.calculate_rsi(prices)
        self.assertIsInstance(rsi, float)
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
    
    def test_calculate_rsi_insufficient_data(self):
        """Test RSI with insufficient data"""
        prices = [100, 102]
        rsi = self.analyzer.calculate_rsi(prices)
        self.assertEqual(rsi, 50.0)
    
    def test_calculate_macd(self):
        """Test MACD calculation"""
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120, 119, 121, 123, 122, 124, 126]
        macd, signal, histogram = self.analyzer.calculate_macd(prices)
        
        self.assertIsInstance(macd, float)
        self.assertIsInstance(signal, float)
        self.assertIsInstance(histogram, float)
    
    def test_calculate_macd_insufficient_data(self):
        """Test MACD with insufficient data"""
        prices = [100, 102]
        macd, signal, histogram = self.analyzer.calculate_macd(prices)
        self.assertEqual(macd, 0.0)
        self.assertEqual(signal, 0.0)
        self.assertEqual(histogram, 0.0)
    
    def test_detect_pattern(self):
        """Test pattern detection"""
        # Bullish trend
        bullish_prices = [100, 102, 104, 106, 108]
        pattern = self.analyzer.detect_pattern(bullish_prices)
        self.assertEqual(pattern, "bullish_trend")
        
        # Bearish trend
        bearish_prices = [108, 106, 104, 102, 100]
        pattern = self.analyzer.detect_pattern(bearish_prices)
        self.assertEqual(pattern, "bearish_trend")
        
        # Insufficient data
        short_prices = [100, 102]
        pattern = self.analyzer.detect_pattern(short_prices)
        self.assertEqual(pattern, "insufficient_data")

class TestSentimentAnalyzer(unittest.TestCase):
    """Test SentimentAnalyzer class"""
    
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_analyze_sentiment_positive(self):
        """Test positive sentiment analysis"""
        sentiment = self.analyzer.analyze_sentiment("This is great news!")
        self.assertGreater(sentiment, 0)
    
    def test_analyze_sentiment_negative(self):
        """Test negative sentiment analysis"""
        sentiment = self.analyzer.analyze_sentiment("This is terrible news!")
        self.assertLess(sentiment, 0)
    
    def test_analyze_sentiment_neutral(self):
        """Test neutral sentiment analysis"""
        sentiment = self.analyzer.analyze_sentiment("This is okay.")
        self.assertAlmostEqual(sentiment, 0, places=1)
    
    def test_calculate_market_sentiment(self):
        """Test market sentiment calculation"""
        news_items = [
            NewsData("Good news", "Content", "Source", 0.5, 0.8, "2024-01-01"),
            NewsData("Bad news", "Content", "Source", -0.3, 0.6, "2024-01-01")
        ]
        sentiment = self.analyzer.calculate_market_sentiment(news_items)
        self.assertIsInstance(sentiment, float)
        self.assertGreaterEqual(sentiment, -1)
        self.assertLessEqual(sentiment, 1)
    
    def test_calculate_market_sentiment_empty(self):
        """Test market sentiment with empty list"""
        sentiment = self.analyzer.calculate_market_sentiment([])
        self.assertEqual(sentiment, 0.0)

class TestCombinedFinancialAgent(unittest.TestCase):
    """Test CombinedFinancialAgent class"""
    
    def setUp(self):
        # Create a temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        
        # Mock the database connection
        with patch('combined_app.sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            
            self.agent = CombinedFinancialAgent()
    
    def tearDown(self):
        # Clean up temporary database
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent.config)
        self.assertIsNotNone(self.agent.cache)
        self.assertIsNotNone(self.agent.technical_analyzer)
        self.assertIsNotNone(self.agent.sentiment_analyzer)
    
    def test_calculate_volatility(self):
        """Test volatility calculation"""
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
        volatility = self.agent.calculate_volatility(prices)
        self.assertIsInstance(volatility, float)
        self.assertGreaterEqual(volatility, 0)
    
    def test_calculate_volatility_insufficient_data(self):
        """Test volatility with insufficient data"""
        prices = [100]
        volatility = self.agent.calculate_volatility(prices)
        self.assertEqual(volatility, 0.0)
    
    def test_assess_risk_level(self):
        """Test risk level assessment"""
        # High risk
        risk_level = self.agent.assess_risk_level(10.0, 85.0, 0.8)
        self.assertEqual(risk_level, "HIGH")
        
        # Medium risk
        risk_level = self.agent.assess_risk_level(3.0, 75.0, 0.3)
        self.assertEqual(risk_level, "MEDIUM")
        
        # Low risk
        risk_level = self.agent.assess_risk_level(1.0, 50.0, 0.1)
        self.assertEqual(risk_level, "LOW")
    
    def test_generate_recommendation(self):
        """Test recommendation generation"""
        market_data = MarketData("AAPL", "2024-01-01", 150.0, 152.0, 155.0, 148.0, 1000000, 1.33)
        
        # Test different scenarios
        rec1 = self.agent.generate_recommendation(market_data, 25.0, 0.5, 0.3, 2.0)
        self.assertIn(rec1, ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"])
        
        rec2 = self.agent.generate_recommendation(market_data, 75.0, -0.5, -0.3, 8.0)
        self.assertIn(rec2, ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"])
    
    def test_calculate_target_price(self):
        """Test target price calculation"""
        market_data = MarketData("AAPL", "2024-01-01", 150.0, 152.0, 155.0, 148.0, 1000000, 1.33)
        
        target_buy = self.agent.calculate_target_price(market_data, "BUY")
        self.assertGreater(target_buy, market_data.close_price)
        
        target_hold = self.agent.calculate_target_price(market_data, "HOLD")
        self.assertGreater(target_hold, market_data.close_price)
        
        target_sell = self.agent.calculate_target_price(market_data, "SELL")
        self.assertLess(target_sell, market_data.close_price)
    
    def test_calculate_stop_loss(self):
        """Test stop loss calculation"""
        market_data = MarketData("AAPL", "2024-01-01", 150.0, 152.0, 155.0, 148.0, 1000000, 1.33)
        
        stop_high = self.agent.calculate_stop_loss(market_data, "HIGH")
        stop_medium = self.agent.calculate_stop_loss(market_data, "MEDIUM")
        stop_low = self.agent.calculate_stop_loss(market_data, "LOW")
        
        self.assertLess(stop_high, market_data.close_price)
        self.assertLess(stop_medium, market_data.close_price)
        self.assertLess(stop_low, market_data.close_price)
        self.assertLess(stop_high, stop_medium)
        self.assertLess(stop_medium, stop_low)
    
    def test_determine_trend(self):
        """Test trend determination"""
        # Uptrend
        uptrend_prices = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]
        trend = self.agent.determine_trend(uptrend_prices)
        self.assertEqual(trend, "UPTREND")
        
        # Downtrend
        downtrend_prices = [118, 116, 114, 112, 110, 108, 106, 104, 102, 100]
        trend = self.agent.determine_trend(downtrend_prices)
        self.assertEqual(trend, "DOWNTREND")
        
        # Insufficient data
        short_prices = [100, 102]
        trend = self.agent.determine_trend(short_prices)
        self.assertEqual(trend, "UNKNOWN")

class TestAPIEndpoints(unittest.TestCase):
    """Test FastAPI endpoints"""
    
    def setUp(self):
        # Initialize the agent for testing
        init_agent()
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint returns HTML"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Combined Financial Analysis Platform", response.text)
    
    def test_dashboard_endpoint(self):
        """Test dashboard endpoint returns HTML"""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Financial Analysis Dashboard", response.text)
    
    def test_events_endpoint(self):
        """Test events endpoint"""
        response = self.client.get("/events")
        self.assertEqual(response.status_code, 200)
        self.assertIn("TechSum Events Metrics", response.text)
    
    def test_aapl_endpoint(self):
        """Test AAPL endpoint"""
        response = self.client.get("/aapl")
        self.assertEqual(response.status_code, 200)
        self.assertIn("AAPL Stock Metrics", response.text)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
    
    @patch('combined_app.agent.perform_comprehensive_analysis')
    def test_analyze_endpoint(self, mock_analysis):
        """Test analyze endpoint"""
        # Mock the analysis result
        mock_result = AnalysisResult(
            symbol="AAPL",
            current_price=152.0,
            trend="UPTREND",
            confidence=0.75,
            risk_level="MEDIUM",
            recommendation="BUY",
            target_price=175.0,
            stop_loss=130.0,
            sentiment_score=0.2,
            technical_indicators={"rsi": 65.0, "macd": 0.5},
            reasoning="Test reasoning",
            timestamp="2024-01-01T00:00:00"
        )
        mock_analysis.return_value = mock_result
        
        response = self.client.get("/api/analyze/AAPL")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["symbol"], "AAPL")
        self.assertEqual(data["recommendation"], "BUY")
    
    def test_analyze_endpoint_invalid_symbol(self):
        """Test analyze endpoint with invalid symbol"""
        response = self.client.get("/api/analyze/")
        self.assertEqual(response.status_code, 404)
    
    @patch('combined_app.fetch_alpha_vantage_stock')
    def test_stock_endpoint(self, mock_fetch):
        """Test stock data endpoint"""
        mock_fetch.return_value = {"status": "success", "data": {"test": "data"}}
        
        response = self.client.get("/stock/AAPL")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
    
    @patch('combined_app.fetch_news_data')
    def test_news_endpoint(self, mock_fetch):
        """Test news data endpoint"""
        mock_fetch.return_value = {"status": "success", "data": {"articles": []}}
        
        response = self.client.get("/news/AAPL")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete application"""
    
    def setUp(self):
        self.client = TestClient(app)
        init_agent()
    
    def test_full_workflow(self):
        """Test complete workflow from request to response"""
        # Test that the application starts without errors
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        
        # Test health check
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        
        # Test that all main endpoints are accessible
        endpoints = ["/", "/dashboard", "/events", "/aapl", "/api/health"]
        for endpoint in endpoints:
            response = self.client.get(endpoint)
            self.assertEqual(response.status_code, 200, f"Endpoint {endpoint} failed")
    
    def test_error_handling(self):
        """Test error handling across the application"""
        # Test non-existent endpoint
        response = self.client.get("/nonexistent")
        self.assertEqual(response.status_code, 404)
        
        # Test invalid API calls
        response = self.client.get("/api/analyze/")
        self.assertEqual(response.status_code, 404)

class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def setUp(self):
        self.client = TestClient(app)
        init_agent()
    
    def test_response_times(self):
        """Test that endpoints respond within reasonable time"""
        import time
        
        endpoints = ["/", "/dashboard", "/events", "/aapl", "/api/health"]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = self.client.get(endpoint)
            end_time = time.time()
            
            self.assertEqual(response.status_code, 200)
            self.assertLess(end_time - start_time, 5.0, f"Endpoint {endpoint} took too long")

def run_tests():
    """Run all tests and display results"""
    print("Running Comprehensive Test Suite for Combined Financial Analysis Application")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDataStructures,
        TestDataCache,
        TestTechnicalAnalyzer,
        TestSentimentAnalyzer,
        TestCombinedFinancialAgent,
        TestAPIEndpoints,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
