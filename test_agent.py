#!/usr/bin/env python3
"""
Direct test of the AI agent functionality
"""
import asyncio
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our modules
from config import ALPHA_VANTAGE_KEY, NEWS_API_KEY

# Create a simple test
async def test_agent():
    from example2 import FinancialAIAgent
    
    config = {
        'alpha_vantage_key': ALPHA_VANTAGE_KEY,
        'news_api_key': NEWS_API_KEY,
    }
    
    print(f"Testing AI agent with Alpha Vantage key: {ALPHA_VANTAGE_KEY[:10]}...")
    print(f"Testing AI agent with News API key: {NEWS_API_KEY[:10]}...")
    
    agent = FinancialAIAgent(config)
    
    # Test market data fetching
    print("\n=== Testing Market Data ===")
    market_data = await agent.fetch_market_data("AAPL")
    if market_data:
        print(f"✅ Market data fetched successfully!")
        print(f"   Symbol: {market_data.symbol}")
        print(f"   Price: ${market_data.close_price}")
        print(f"   Volume: {market_data.volume:,}")
    else:
        print("❌ Failed to fetch market data")
    
    # Test news data fetching
    print("\n=== Testing News Data ===")
    news_data = await agent.fetch_news_data("AAPL")
    if news_data:
        print(f"✅ News data fetched successfully!")
        print(f"   Found {len(news_data)} news items")
        for item in news_data[:3]:
            print(f"   - {item.title[:60]}...")
    else:
        print("❌ No news data fetched")
    
    # Test full analysis
    print("\n=== Testing Full Analysis ===")
    try:
        result = await agent.perform_comprehensive_analysis("AAPL")
        print(f"✅ Full analysis completed!")
        print(f"   Symbol: {result.symbol}")
        print(f"   Price: ${result.current_price}")
        print(f"   Recommendation: {result.recommendation}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Risk Level: {result.risk_level}")
    except Exception as e:
        print(f"❌ Full analysis failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_agent())