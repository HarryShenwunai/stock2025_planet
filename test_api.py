import requests
from config import ALPHA_VANTAGE_KEY

# Test Alpha Vantage API
symbol = "AAPL"
url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"

print(f"Testing API with key: {ALPHA_VANTAGE_KEY[:10]}...")
print(f"URL: {url}")

response = requests.get(url)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")