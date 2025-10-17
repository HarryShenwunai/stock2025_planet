"""Simple import test to check if all modules load correctly"""
import sys

print("Testing imports...")

try:
    print("1. Testing data_structures...")
    from data_structures import MarketData, NewsData, AnalysisResult, DataCache
    print("   ✓ data_structures OK")
except Exception as e:
    print(f"   ✗ data_structures FAILED: {e}")
    sys.exit(1)

try:
    print("2. Testing technical_analysis...")
    from technical_analysis import TechnicalAnalyzer
    print("   ✓ technical_analysis OK")
except Exception as e:
    print(f"   ✗ technical_analysis FAILED: {e}")
    sys.exit(1)

try:
    print("3. Testing api_client...")
    from api_client import get_api_config
    print("   ✓ api_client OK")
except Exception as e:
    print(f"   ✗ api_client FAILED: {e}")
    sys.exit(1)

try:
    print("4. Testing sentiment_analysis...")
    from sentiment_analysis import SentimentAnalyzer
    print("   ✓ sentiment_analysis OK")
except Exception as e:
    print(f"   ✗ sentiment_analysis FAILED: {e}")
    sys.exit(1)

try:
    print("5. Testing financial_glossary...")
    from financial_glossary import FINANCIAL_TERMS
    print("   ✓ financial_glossary OK")
except Exception as e:
    print(f"   ✗ financial_glossary FAILED: {e}")
    sys.exit(1)

try:
    print("6. Testing main app import...")
    from main import app
    print("   ✓ main app OK")
    print("\n✓ All imports successful!")
except Exception as e:
    print(f"   ✗ main app FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

