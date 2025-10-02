from typing import Dict, Any
import requests
from fastapi import FastAPI, HTTPException
from datetime import datetime
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def index():
    """Welcome page with API documentation"""
    return """
    <html>
        <head>
            <title>AI Financial Agent API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; }
                h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                ul { list-style-type: none; padding: 0; }
                li { margin: 10px 0; padding: 10px; background: #ecf0f1; border-radius: 5px; }
                a { color: #3498db; text-decoration: none; font-weight: bold; }
                a:hover { color: #2980b9; }
                .endpoint { font-family: monospace; background: #2c3e50; color: white; padding: 5px 10px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Financial Agent API</h1>
                <h2>Available Endpoints:</h2>
                <ul>
                    <li><span class="endpoint">GET /api/analyze/{symbol}</span> - Complete stock analysis</li>
                    <li><span class="endpoint">GET /api/history/{symbol}</span> - Historical analysis data</li>
                    <li><span class="endpoint">GET /api/health</span> - System health check</li>
                    <li><span class="endpoint">GET /docs</span> - Interactive API documentation</li>
                    <li><span class="endpoint">GET /redoc</span> - Alternative API documentation</li>
                </ul>
                <h2>Example Usage:</h2>
                <ul>
                    <li><a href="/api/analyze/AAPL">Analyze AAPL</a></li>
                    <li><a href="/api/analyze/TSLA">Analyze TSLA</a></li>
                    <li><a href="/api/analyze/MSFT">Analyze MSFT</a></li>
                    <li><a href="/api/health">Health Check</a></li>
                    <li><a href="/docs">API Documentation</a></li>
                </ul>
                <p><em> Tip: Visit <a href="/docs">/docs</a> for interactive API testing!</em></p>
            </div>
        </body>
    </html>
    """

@app.get("/stock/{symbol}")
def get_stock_data(symbol: str) -> Dict[str, Any]:
    """Get stock data using Alpha Vantage API"""
    try:
        # Load API key from config
        try:
            from config import ALPHA_VANTAGE_KEY
            api_key = ALPHA_VANTAGE_KEY
        except ImportError:
            print("Warning: alphavantage API key was not found")
        
        # Fetch stock data
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

@app.get("/news/{symbol}")
def get_news_data(symbol: str) -> Dict[str, Any]:
    """Get news data using News API"""
    try:
        try:
            from config import NEWS_API_KEY
            api_key = NEWS_API_KEY
        except ImportError:
            print("Warning: news API key was not found")
        
        url = f"https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&pageSize=10&apiKey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news data: {str(e)}")
    
if __name__ == "__main__":
    print("AI Financial Agent - Starting FastAPI Server")
    print("=" * 50)
    print("Available endpoints:")
    print("  GET / - Welcome page")
    print("  GET /api/analyze/{symbol} - Analyze stock")
    print("  GET /api/history/{symbol} - Analysis history")
    print("  GET /api/health - Health check")
    print("  GET /docs - Interactive API documentation")
    print("  GET /redoc - Alternative API documentation")
    print("=" * 50)
    
    # Start the FastAPI server with uvicorn
    import uvicorn
    print("\n Starting FastAPI server on http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")