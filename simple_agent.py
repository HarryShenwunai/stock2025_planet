#!/usr/bin/env python3
"""
Simple AI Financial Agent Test
"""
import requests
from flask import Flask, jsonify
from config import ALPHA_VANTAGE_KEY

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <h1>Simple AI Financial Agent</h1>
    <h2>Test Endpoints:</h2>
    <ul>
        <li><a href="/test/AAPL">Test AAPL</a></li>
        <li><a href="/api/AAPL">Full Analysis AAPL</a></li>
    </ul>
    '''

@app.route('/test/<symbol>')
def test_symbol(symbol):
    """Simple test endpoint"""
    try:
        # Test Alpha Vantage API
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Global Quote" in data:
            quote = data["Global Quote"]
            return jsonify({
                "status": "success",
                "symbol": symbol,
                "price": quote.get("05. price"),
                "change": quote.get("09. change"),
                "change_percent": quote.get("10. change percent"),
                "volume": quote.get("06. volume")
            })
        else:
            return jsonify({
                "status": "error",
                "message": "No quote data found",
                "raw_response": data
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/<symbol>')
def analyze_symbol(symbol):
    """Simple analysis"""
    try:
        # Get basic stock data
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Global Quote" in data:
            quote = data["Global Quote"]
            price = float(quote.get("05. price", 0))
            change_percent = float(quote.get("10. change percent", "0%").rstrip("%"))
            
            # Simple analysis logic
            if change_percent > 2:
                recommendation = "STRONG_BUY"
                confidence = 0.8
            elif change_percent > 0:
                recommendation = "BUY"
                confidence = 0.6
            elif change_percent > -2:
                recommendation = "HOLD" 
                confidence = 0.5
            else:
                recommendation = "SELL"
                confidence = 0.7
                
            return jsonify({
                "symbol": symbol,
                "current_price": price,
                "change_percent": change_percent,
                "recommendation": recommendation,
                "confidence": confidence,
                "risk_level": "MEDIUM" if abs(change_percent) < 3 else "HIGH",
                "analysis_summary": f"{symbol} shows {'bullish' if change_percent > 0 else 'bearish'} momentum with {abs(change_percent):.2f}% change",
                "timestamp": response.headers.get('date', 'N/A')
            })
        else:
            return jsonify({"error": "Could not fetch stock data", "raw_response": data}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Simple AI Financial Agent")
    print("Available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)