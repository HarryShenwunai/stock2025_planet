#!/usr/bin/env python3
"""
Simple test runner for combined_app.py
=====================================

This script runs basic tests for the combined financial analysis application.

Usage:
    python simple_test.py
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        from combined_app import (
            MarketData, NewsData, AnalysisResult,
            DataCache, TechnicalAnalyzer, SentimentAnalyzer,
            CombinedFinancialAgent, app
        )
        print("SUCCESS: All imports successful")
        return True
    except Exception as e:
        print(f"FAILED: Import error - {e}")
        return False

def test_data_structures():
    """Test data structure creation"""
    print("Testing data structures...")
    try:
        from combined_app import MarketData, NewsData, AnalysisResult
        
        # Test MarketData
        market_data = MarketData(
            symbol="TEST",
            timestamp="2024-01-01T00:00:00",
            open_price=100.0,
            close_price=102.0,
            high_price=105.0,
            low_price=98.0,
            volume=1000000,
            change_percent=2.0
        )
        assert market_data.symbol == "TEST"
        assert market_data.close_price == 102.0
        
        # Test NewsData
        news_data = NewsData(
            title="Test News",
            content="Test content",
            source="Test Source",
            sentiment_score=0.5,
            relevance_score=0.8,
            timestamp="2024-01-01T00:00:00"
        )
        assert news_data.sentiment_score == 0.5
        
        print("SUCCESS: Data structures working")
        return True
    except Exception as e:
        print(f"FAILED: Data structure error - {e}")
        return False

def test_cache():
    """Test cache functionality"""
    print("Testing cache...")
    try:
        from combined_app import DataCache
        
        cache = DataCache()
        cache.set("test_key", "test_value")
        assert cache.get("test_key") == "test_value"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
        
        print("SUCCESS: Cache working")
        return True
    except Exception as e:
        print(f"FAILED: Cache error - {e}")
        return False

def test_technical_analysis():
    """Test technical analysis functions"""
    print("Testing technical analysis...")
    try:
        from combined_app import TechnicalAnalyzer
        
        analyzer = TechnicalAnalyzer()
        
        # Test RSI calculation
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
        rsi = analyzer.calculate_rsi(prices)
        assert 0 <= rsi <= 100
        
        # Test MACD calculation
        macd, signal, histogram = analyzer.calculate_macd(prices)
        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(histogram, float)
        
        # Test pattern detection
        bullish_prices = [100, 102, 104, 106, 108]
        pattern = analyzer.detect_pattern(bullish_prices)
        assert pattern in ["bullish_trend", "bearish_trend", "sideways", "head_shoulders", "insufficient_data"]
        
        print("SUCCESS: Technical analysis working")
        return True
    except Exception as e:
        print(f"FAILED: Technical analysis error - {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis"""
    print("Testing sentiment analysis...")
    try:
        from combined_app import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        
        # Test sentiment analysis
        positive_sentiment = analyzer.analyze_sentiment("This is great news!")
        negative_sentiment = analyzer.analyze_sentiment("This is terrible news!")
        neutral_sentiment = analyzer.analyze_sentiment("This is okay.")
        
        assert isinstance(positive_sentiment, float)
        assert isinstance(negative_sentiment, float)
        assert isinstance(neutral_sentiment, float)
        
        print("SUCCESS: Sentiment analysis working")
        return True
    except Exception as e:
        print(f"FAILED: Sentiment analysis error - {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("Testing API endpoints...")
    try:
        from fastapi.testclient import TestClient
        from combined_app import app, init_agent
        
        # Initialize agent
        init_agent()
        
        # Create test client
        client = TestClient(app)
        
        # Test main endpoints
        endpoints = [
            ("/", "Main page"),
            ("/dashboard", "Dashboard"),
            ("/events", "Events"),
            ("/aapl", "AAPL page"),
            ("/api/health", "Health check")
        ]
        
        for endpoint, description in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200, f"Endpoint {endpoint} failed with status {response.status_code}"
        
        print("SUCCESS: API endpoints working")
        return True
    except Exception as e:
        print(f"FAILED: API endpoints error - {e}")
        return False

def test_agent_functionality():
    """Test AI agent functionality"""
    print("Testing AI agent...")
    try:
        from combined_app import CombinedFinancialAgent, MarketData
        from unittest.mock import patch
        
        # Mock database connection
        with patch('combined_app.sqlite3.connect'):
            agent = CombinedFinancialAgent()
            
            # Test volatility calculation
            prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
            volatility = agent.calculate_volatility(prices)
            assert isinstance(volatility, float)
            assert volatility >= 0
            
            # Test risk assessment
            risk_level = agent.assess_risk_level(volatility, 50.0, 0.1)
            assert risk_level in ["LOW", "MEDIUM", "HIGH"]
            
            # Test recommendation generation
            market_data = MarketData("TEST", "2024-01-01", 100.0, 102.0, 105.0, 98.0, 1000000, 2.0)
            recommendation = agent.generate_recommendation(market_data, 50.0, 0.5, 0.1, 2.0)
            assert recommendation in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
            
            # Test target price calculation
            target_price = agent.calculate_target_price(market_data, recommendation)
            assert isinstance(target_price, float)
            assert target_price > 0
            
            # Test stop loss calculation
            stop_loss = agent.calculate_stop_loss(market_data, "MEDIUM")
            assert isinstance(stop_loss, float)
            assert stop_loss > 0
            
            # Test trend determination
            trend = agent.determine_trend(prices)
            assert trend in ["UPTREND", "DOWNTREND", "SIDEWAYS", "UNKNOWN"]
        
        print("SUCCESS: AI agent working")
        return True
    except Exception as e:
        print(f"FAILED: AI agent error - {e}")
        return False

def main():
    """Run all tests"""
    print("Combined Financial Analysis Application - Test Suite")
    print("=" * 60)
    
    # Check if combined_app.py exists
    if not os.path.exists("combined_app.py"):
        print("ERROR: combined_app.py not found in current directory")
        return False
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Data Structures", test_data_structures),
        ("Cache Functionality", test_cache),
        ("Technical Analysis", test_technical_analysis),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("API Endpoints", test_api_endpoints),
        ("AI Agent", test_agent_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print('-' * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"CRASHED: {test_name} - {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print('=' * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! The application is working correctly.")
        print("\nTo run the application:")
        print("python combined_app.py")
        print("\nThen visit: http://localhost:8000")
    else:
        print(f"\n{total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
