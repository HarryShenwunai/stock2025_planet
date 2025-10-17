"""Simple import test to check if all modules load correctly"""
import sys

print("Testing imports...")

try:
    print("1. Testing data_structures...")
    from data_structures import MarketData, NewsData, AnalysisResult, DataCache
    print("   [OK] data_structures")
except Exception as e:
    print(f"   [FAILED] data_structures: {e}")
    sys.exit(1)

try:
    print("2. Testing technical_analysis...")
    from technical_analysis import TechnicalAnalyzer
    print("   [OK] technical_analysis")
except Exception as e:
    print(f"   [FAILED] technical_analysis: {e}")
    sys.exit(1)

try:
    print("3. Testing api_client...")
    from api_client import get_api_config
    print("   [OK] api_client")
except Exception as e:
    print(f"   [FAILED] api_client: {e}")
    sys.exit(1)

try:
    print("4. Testing sentiment_analysis...")
    from sentiment_analysis import SentimentAnalyzer
    print("   [OK] sentiment_analysis")
except Exception as e:
    print(f"   [FAILED] sentiment_analysis: {e}")
    sys.exit(1)

try:
    print("5. Testing financial_glossary...")
    from financial_glossary import FINANCIAL_TERMS
    print("   [OK] financial_glossary")
except Exception as e:
    print(f"   [FAILED] financial_glossary: {e}")
    sys.exit(1)

try:
    print("6. Testing main app import...")
    from main import app
    print("   [OK] main app")
    print("\n[SUCCESS] All imports successful!")
except Exception as e:
    print(f"   [FAILED] main app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

