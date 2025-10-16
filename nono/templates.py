import json
from fastapi.responses import HTMLResponse
from api_client import fetch_stock_data
# ===== HTML TEMPLATES =====
def get_main_page_html():
    """Main dashboard HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Combined Financial Analysis Platform</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { 
                background: rgba(255,255,255,0.95); 
                border-radius: 15px; 
                padding: 30px; 
                margin-bottom: 30px; 
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .header h1 { 
                color: #2c3e50; 
                font-size: 2.5em; 
                margin-bottom: 10px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .header p { color: #7f8c8d; font-size: 1.2em; }
            .sections { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px; 
            }
            .section { 
                background: rgba(255,255,255,0.95); 
                border-radius: 15px; 
                padding: 25px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .section:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.15);
            }
            .section h2 { 
                color: #2c3e50; 
                margin-bottom: 15px; 
                font-size: 1.5em;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            .endpoints { list-style: none; }
            .endpoints li { 
                margin: 12px 0; 
                padding: 12px; 
                background: #f8f9fa; 
                border-radius: 8px; 
                border-left: 4px solid #3498db;
                transition: background 0.3s ease;
            }
            .endpoints li:hover { background: #e3f2fd; }
            .endpoint { 
                font-family: 'Courier New', monospace; 
                background: #2c3e50; 
                color: white; 
                padding: 6px 12px; 
                border-radius: 5px; 
                font-size: 0.9em;
                display: inline-block;
                margin-bottom: 5px;
            }
            .method-get { background: #27ae60 !important; }
            .method-post { background: #e74c3c !important; }
            a { 
                color: #3498db; 
                text-decoration: none; 
                font-weight: 600;
                transition: color 0.3s ease;
            }
            a:hover { color: #2980b9; }
            .demo-section {
                background: rgba(255,255,255,0.95);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .demo-links {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            .demo-link {
                background: linear-gradient(45deg, #3498db, #2980b9);
                color: white !important;
                padding: 15px 20px;
                border-radius: 10px;
                text-align: center;
                font-weight: bold;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                text-decoration: none;
            }
            .demo-link:hover {
                transform: scale(1.05);
                box-shadow: 0 5px 20px rgba(52, 152, 219, 0.4);
                color: white !important;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Combined Financial Analysis Platform</h1>
                <p>Unified platform combining Flask, FastAPI, and AI-powered financial analysis</p>
            </div>
            
            <div class="sections">
                <div class="section">
                    <h2>Example 1 - Events & Stock Data</h2>
                    <ul class="endpoints">
                        <li><span class="endpoint method-get">GET /events</span><br>TechSum events and metrics display</li>
                        <li><span class="endpoint method-get">GET /aapl</span><br>AAPL stock metrics with visualization</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Example 2 - AI Financial Agent</h2>
                    <ul class="endpoints">
                        <li><span class="endpoint method-get">GET /api/analyze/{symbol}</span><br>Comprehensive AI stock analysis</li>
                        <li><span class="endpoint method-get">GET /api/history/{symbol}</span><br>Historical analysis data</li>
                        <li><span class="endpoint method-get">GET /api/health</span><br>System health check</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Example 3 - Stock & News APIs</h2>
                    <ul class="endpoints">
                        <li><span class="endpoint method-get">GET /stock/{symbol}</span><br>Alpha Vantage stock data</li>
                        <li><span class="endpoint method-get">GET /news/{symbol}</span><br>News API data with sentiment</li>
                    </ul>
                </div>
            </div>
            
            <div class="demo-section">
                <h2>Quick Demo Links</h2>
                <div class="demo-links">
                    <a href="/events" class="demo-link">TechSum Events</a>
                    <a href="/aapl" class="demo-link">AAPL Analysis</a>
                    <a href="/api/analyze/TSLA" class="demo-link">TSLA AI Analysis</a>
                    <a href="/stock/MSFT" class="demo-link">MSFT Stock Data</a>
                    <a href="/news/AAPL" class="demo-link">AAPL News</a>
                    <a href="/api/health" class="demo-link">Health Check</a>
                    <a href="/docs" class="demo-link">API Docs</a>
                    <a href="/dashboard" class="demo-link">Dashboard</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

def get_dashboard_html():
    """Interactive dashboard HTML"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: #f5f7fa;
                color: #333;
            }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .header { 
                background: white; 
                border-radius: 10px; 
                padding: 20px; 
                margin-bottom: 20px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .search-section { 
                display: flex; 
                gap: 10px; 
                align-items: center; 
            }
            input[type="text"] {
                padding: 10px 15px;
                border: 2px solid #e1e8ed;
                border-radius: 8px;
                font-size: 16px;
                width: 150px;
                transition: border-color 0.3s ease;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #3498db;
            }
            button {
                background: linear-gradient(45deg, #3498db, #2980b9);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: transform 0.2s ease;
            }
            button:hover { transform: scale(1.05); }
            .dashboard-grid { 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 20px; 
                margin-bottom: 20px; 
            }
            .widget { 
                background: white; 
                border-radius: 10px; 
                padding: 20px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                position: relative;
            }
            .widget h3 { 
                margin-bottom: 15px; 
                color: #2c3e50; 
                border-bottom: 2px solid #3498db; 
                padding-bottom: 10px; 
            }
            /* Ensure the price chart has a stable height to prevent growth */
            .widget.chart-widget {
                height: 420px;
            }
            #priceChart {
                height: 100% !important;
                width: 100% !important;
                display: block;
            }
            .analysis-result { 
                grid-column: 1 / -1; 
                background: white; 
                border-radius: 10px; 
                padding: 20px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px; 
            }
            .metrics-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
                gap: 15px; 
                margin-top: 15px; 
            }
            .metric { 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 8px; 
                text-align: center; 
                border-left: 4px solid #3498db; 
            }
            .metric-value { 
                font-size: 1.5em; 
                font-weight: bold; 
                color: #2c3e50; 
            }
            .metric-label { 
                font-size: 0.9em; 
                color: #7f8c8d; 
                margin-top: 5px; 
            }
            .recommendation { 
                padding: 15px; 
                border-radius: 8px; 
                margin: 15px 0; 
                font-weight: bold; 
                text-align: center; 
            }
            .rec-buy { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .rec-sell { background: #f8d7da; color: #721c24; border: 1px solid #f1b0b7; }
            .rec-hold { background: #fff3cd; color: #856404; border: 1px solid #ffd700; }
            .loading { text-align: center; padding: 20px; color: #7f8c8d; }
            .navigation { 
                background: white; 
                border-radius: 10px; 
                padding: 15px; 
                margin-bottom: 20px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }
            .nav-link { 
                color: #3498db; 
                text-decoration: none; 
                margin-right: 20px; 
                font-weight: 600; 
                padding: 8px 16px;
                border-radius: 5px;
                transition: background 0.3s ease;
            }
            .nav-link:hover { 
                background: #e3f2fd; 
                color: #2980b9; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Financial Analysis Dashboard</h1>
                <div class="search-section">
                    <input type="text" id="symbolInput" placeholder="AAPL" value="AAPL">
                    <button onclick="analyzeStock()">Analyze</button>
                </div>
            </div>
            
            <div class="navigation">
                <a href="/" class="nav-link">Home</a>
                <a href="/events" class="nav-link">Events</a>
                <a href="/docs" class="nav-link">API Docs</a>
            </div>
            
            <div id="analysisResult" class="analysis-result" style="display: none;">
                <h3>AI Analysis Result</h3>
                <div id="analysisContent"></div>
            </div>
            
            <div class="dashboard-grid">
                <div class="widget chart-widget">
                    <h3>Price Chart</h3>
                    <canvas id="priceChart"></canvas>
                </div>
                
                <div class="widget">
                    <h3>Technical Indicators</h3>
                    <div id="technicalIndicators">
                        <p class="loading">Load a symbol to see technical indicators</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let priceChart = null;
            
            async function analyzeStock() {
                const symbol = document.getElementById('symbolInput').value.toUpperCase();
                if (!symbol) return;
                
                // Show loading
                const resultDiv = document.getElementById('analysisResult');
                const contentDiv = document.getElementById('analysisContent');
                resultDiv.style.display = 'block';
                contentDiv.innerHTML = '<div class="loading">Analyzing ' + symbol + '...</div>';
                
                try {
                    // Fetch analysis
                    const response = await fetch(`/api/analyze/${symbol}`);
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayAnalysis(data);
                        updateTechnicalIndicators(data.technical_indicators);
                        createPriceChart(data);
                    } else {
                        contentDiv.innerHTML = '<div class="loading">Error: ' + (data.detail || 'Unknown error') + '</div>';
                    }
                } catch (error) {
                    contentDiv.innerHTML = '<div class="loading">Network error: ' + error.message + '</div>';
                }
            }
            
            function displayAnalysis(data) {
                const contentDiv = document.getElementById('analysisContent');
                
                let recClass = 'rec-hold';
                if (data.recommendation.includes('BUY')) recClass = 'rec-buy';
                else if (data.recommendation.includes('SELL')) recClass = 'rec-sell';
                
                contentDiv.innerHTML = `
                    <div class="recommendation ${recClass}">
                        ${data.recommendation} - ${data.symbol}
                    </div>
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value">$${data.current_price.toFixed(2)}</div>
                            <div class="metric-label">Current Price</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.trend}</div>
                            <div class="metric-label">Trend</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${(data.confidence * 100).toFixed(0)}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${data.risk_level}</div>
                            <div class="metric-label">Risk Level</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">$${data.target_price.toFixed(2)}</div>
                            <div class="metric-label">Target Price</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">$${data.stop_loss.toFixed(2)}</div>
                            <div class="metric-label">Stop Loss</div>
                        </div>
                    </div>
                    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                        <h4>Analysis Reasoning:</h4>
                        <p style="margin-top: 10px; line-height: 1.6;">${data.reasoning}</p>
                    </div>
                `;
            }
            
            function updateTechnicalIndicators(indicators) {
                const div = document.getElementById('technicalIndicators');
                div.innerHTML = `
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value">${indicators.rsi.toFixed(1)}</div>
                            <div class="metric-label">RSI</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${indicators.macd.toFixed(3)}</div>
                            <div class="metric-label">MACD</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">${indicators.volatility.toFixed(2)}%</div>
                            <div class="metric-label">Volatility</div>
                        </div>
                    </div>
                `;
            }
            
            function createPriceChart(data) {
                const ctx = document.getElementById('priceChart').getContext('2d');
                
                // Destroy existing chart
                if (priceChart) priceChart.destroy();
                
                // Generate sample price data for visualization
                const days = 30;
                const dates = [];
                const prices = [];
                const currentPrice = data.current_price;
                
                for (let i = days; i >= 0; i--) {
                    const date = new Date();
                    date.setDate(date.getDate() - i);
                    dates.push(date.toLocaleDateString());
                    
                    // Generate realistic price variations
                    const variation = (Math.random() - 0.5) * 0.05; // ±2.5% variation
                    const price = currentPrice * (1 + variation * (i / days));
                    prices.push(price);
                }
                
                priceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: dates,
                        datasets: [{
                            label: data.symbol + ' Price',
                            data: prices,
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        resizeDelay: 150,
                        plugins: {
                            legend: {
                                display: true
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: false,
                                title: {
                                    display: true,
                                    text: 'Price ($)'
                                }
                            }
                        }
                    }
                });
            }
            
            // Auto-analyze AAPL on load
            window.onload = function() {
                analyzeStock();
            };
        </script>
    </body>
    </html>
    """

async def get_aapl_page_html():
    """AAPL stock page (from example1.py)"""
    url = "https://dataserver.datasum.ai/stock-info/api/v1/stock?symbol=AAPL"
    stock_data = fetch_stock_data(url)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AAPL Stock Metrics</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f8f8f8; margin: 20px; }}
            .page-header {{ background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .page-title {{ font-size: 1.5em; font-weight: bold; color: #333; }}
            .navigation {{ background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .nav-link {{ color: #0077cc; text-decoration: none; margin-right: 20px; font-weight: bold; }}
            .stock-summary {{ background: #fff; border-radius: 10px; margin: 20px 0; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .chart-container {{ background: #fff; border-radius: 10px; margin: 20px 0; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); height: 420px; }}
            #priceChart { width: 100% !important; height: 100% !important; display: block; }
        </style>
    </head>
    <body>
        <div class="page-header">
            <div class="page-title">AAPL Stock Metrics</div>
        </div>
        
        <div class="navigation">
            <a href="/" class="nav-link">Home</a>
            <a href="/events" class="nav-link">Events</a>
            <a href="/aapl" class="nav-link">AAPL Stock</a>
            <a href="/dashboard" class="nav-link">Dashboard</a>
        </div>
        
        <div class="chart-container">
            <h3>AAPL Price Chart</h3>
            <canvas id="priceChart"></canvas>
        </div>
        
        <div class="stock-summary">
            <h3>Stock Information</h3>
            <p>Symbol: AAPL</p>
            <p>Data fetched from: {url}</p>
            <pre>{json.dumps(stock_data, indent=2)[:500]}...</pre>
        </div>
        
        <script>
            // Simple chart example
            const ctx = document.getElementById('priceChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'],
                    datasets: [{{
                        label: 'AAPL Price',
                        data: [150, 152, 148, 155, 153],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    resizeDelay: 150
                }}
            }});
        </script>
    </body>
    </html>
    """
    return html