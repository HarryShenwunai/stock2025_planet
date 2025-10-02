# Configuration file for AI Financial Agent
# Add your real API keys here

# API Keys (get these from the respective services)
ALPHA_VANTAGE_KEY = "CXENBGF94W07652L"  # Replace with your Alpha Vantage API key
NEWS_API_KEY = "0d453906e1f5419d939eb464e65e799f"  # Replace with your News API key

# Database configuration
DATABASE_PATH = "agent_data.db"

# Cache settings
CACHE_TTL_SECONDS = 300  # 5 minutes

# Analysis parameters
DEFAULT_RSI_PERIOD = 14
DEFAULT_HISTORICAL_DAYS = 30
VOLATILITY_THRESHOLD_HIGH = 5.0
VOLATILITY_THRESHOLD_MEDIUM = 2.0

# Risk assessment thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_EXTREME_OVERBOUGHT = 80
RSI_EXTREME_OVERSOLD = 20

# Price target multipliers
STRONG_BUY_TARGET = 1.15
BUY_TARGET = 1.10
HOLD_TARGET = 1.05
SELL_TARGET = 0.95
STRONG_SELL_TARGET = 0.90

# Stop loss multipliers by risk level
STOP_LOSS_HIGH_RISK = 0.90
STOP_LOSS_MEDIUM_RISK = 0.85
STOP_LOSS_LOW_RISK = 0.80

# Sentiment analysis
SENTIMENT_POSITIVE_THRESHOLD = 0.2
SENTIMENT_NEGATIVE_THRESHOLD = -0.2

# Volume threshold for confidence calculation
HIGH_VOLUME_THRESHOLD = 1000000

# Flask API configuration
API_HOST = "0.0.0.0"
API_PORT = 5000
DEBUG_MODE = True

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = "ai_agent.log"