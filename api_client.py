import os
import requests
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# ===== API FUNCTIONS =====
def get_api_config():
    """Get API configuration from config file or environment"""
    try:
        from config import ALPHA_VANTAGE_KEY, NEWS_API_KEY
        return {
            'alpha_vantage_key': ALPHA_VANTAGE_KEY,
            'news_api_key': NEWS_API_KEY,
        }
    except ImportError:
        logger.warning("Config file not found, using environment variables")
        return {
            'alpha_vantage_key': os.getenv('ALPHA_VANTAGE_KEY', 'demo'),
            'news_api_key': os.getenv('NEWS_API_KEY', 'demo'),
        }

def fetch_api_data(url: str) -> List[Dict]:
    """Fetch data from API (from example1.py)"""
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data.get('groups', []) or data.get('data', []) or data.get('results', []) or data.get('items', []) or data.get('events', [])
        elif isinstance(data, list):
            return data
        return []
    except Exception as e:
        logger.error(f"Error fetching API data from {url}: {e}")
        return []

def fetch_stock_data(url: str) -> Dict:
    """Fetch stock data from API (from example1.py)"""
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching stock data from {url}: {e}")
        return {}

def fetch_alpha_vantage_stock(symbol: str) -> Dict[str, Any]:
    """Fetch stock data using Alpha Vantage API (from example3.py)"""
    config = get_api_config()
    api_key = config['alpha_vantage_key']
    
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except Exception as e:
        logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
        return {"status": "error", "message": str(e)}

def fetch_news_data(symbol: str) -> Dict[str, Any]:
    """Fetch news data using News API (from example3.py)"""
    config = get_api_config()
    api_key = config['news_api_key']
    
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&pageSize=10&apiKey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except Exception as e:
        logger.error(f"Error fetching news data for {symbol}: {e}")
        return {"status": "error", "message": str(e)}