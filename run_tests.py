#!/usr/bin/env python3
"""
Simple test runner for combined_app.py
=====================================

This script runs the comprehensive test suite and provides
a user-friendly interface for testing the application.

Usage:
    python run_tests.py

Author: Test Runner v1.0
Date: December 2024
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'requests',
        'numpy',
        'pandas',
        'textblob',
        'sqlite3'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sqlite3':
                import sqlite3
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print("pip install fastapi uvicorn requests numpy pandas textblob")
        return False
    
    print("All required packages are installed")
    return True

def run_basic_tests():
    """Run basic functionality tests"""
    print("\nRunning Basic Functionality Tests...")
    print("-" * 50)
    
    try:
        # Test imports
        print("Testing imports...")
        from combined_app import (
            MarketData, NewsData, AnalysisResult,
            DataCache, TechnicalAnalyzer, SentimentAnalyzer,
            CombinedFinancialAgent, app
        )
        print("All imports successful")
        
        # Test data structures
        print("Testing data structures...")
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
        print("MarketData creation successful")
        
        # Test cache
        print("Testing cache functionality...")
        cache = DataCache()
        cache.set("test", "value")
        assert cache.get("test") == "value"
        print("Cache functionality working")
        
        # Test technical analyzer
        print("Testing technical analysis...")
        analyzer = TechnicalAnalyzer()
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
        rsi = analyzer.calculate_rsi(prices)
        assert 0 <= rsi <= 100
        print("Technical analysis working")
        
        # Test sentiment analyzer
        print("Testing sentiment analysis...")
        sentiment_analyzer = SentimentAnalyzer()
        sentiment = sentiment_analyzer.analyze_sentiment("This is great!")
        assert isinstance(sentiment, float)
        print("Sentiment analysis working")
        
        print("\nAll basic tests passed!")
        return True
        
    except Exception as e:
        print(f"Basic test failed: {e}")
        return False

def run_api_tests():
    """Run API endpoint tests"""
    print("\nRunning API Tests...")
    print("-" * 50)
    
    try:
        from fastapi.testclient import TestClient
        from combined_app import app, init_agent
        
        # Initialize agent
        init_agent()
        
        # Create test client
        client = TestClient(app)
        
        # Test endpoints
        endpoints = [
            ("/", "Main page"),
            ("/dashboard", "Dashboard"),
            ("/events", "Events"),
            ("/aapl", "AAPL page"),
            ("/api/health", "Health check")
        ]
        
        for endpoint, description in endpoints:
            print(f"Testing {description} ({endpoint})...")
            response = client.get(endpoint)
            assert response.status_code == 200, f"Endpoint {endpoint} returned {response.status_code}"
            print(f"✅ {description} working")
        
        print("\n✅ All API tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def run_comprehensive_tests():
    """Run the full test suite"""
    print("\n🔬 Running Comprehensive Test Suite...")
    print("-" * 50)
    
    try:
        # Import and run the test suite
        from test_combined_app import run_tests
        success = run_tests()
        
        if success:
            print("\n✅ Comprehensive test suite passed!")
        else:
            print("\n❌ Some tests failed!")
        
        return success
        
    except Exception as e:
        print(f"❌ Comprehensive test suite failed: {e}")
        return False

def run_application_test():
    """Test the application startup"""
    print("\n🚀 Testing Application Startup...")
    print("-" * 50)
    
    try:
        # Test if the application can be imported and initialized
        from combined_app import app, init_agent
        
        # Initialize the agent
        init_agent()
        
        # Test that the app is properly configured
        assert app is not None
        assert app.title == "Combined Financial Analysis API"
        
        print("✅ Application startup successful")
        print("✅ FastAPI app configured correctly")
        print("✅ Agent initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Application startup test failed: {e}")
        return False

def main():
    """Main test runner function"""
    print("Combined Financial Analysis Application Test Suite")
    print("=" * 60)
    print("Testing combined_app.py functionality...")
    
    # Check if we're in the right directory
    if not os.path.exists("combined_app.py"):
        print("❌ combined_app.py not found in current directory")
        print("Please run this script from the directory containing combined_app.py")
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Run tests in sequence
    tests = [
        ("Basic Functionality", run_basic_tests),
        ("Application Startup", run_application_test),
        ("API Endpoints", run_api_tests),
        ("Comprehensive Suite", run_comprehensive_tests)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_name} Tests")
        print('='*60)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The application is working correctly.")
        print("\nTo run the application:")
        print("python combined_app.py")
        print("\nThen visit: http://localhost:8000")
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
