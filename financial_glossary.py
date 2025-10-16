FINANCIAL_TERMS = {
    "RSI": {
        "name": "Relative Strength Index",
        "description": "A momentum oscillator that measures the speed and magnitude of price changes. Ranges from 0-100.",
        "interpretation": "Above 70 = Overbought, Below 30 = Oversold"
    },
    "MACD": {
        "name": "Moving Average Convergence Divergence",
        "description": "A trend-following momentum indicator that shows the relationship between two moving averages.",
        "interpretation": "Positive = Bullish momentum, Negative = Bearish momentum"
    },
    "VOLATILITY": {
        "name": "Price Volatility",
        "description": "A measure of how much a stock's price fluctuates over time.",
        "interpretation": "High volatility = Higher risk, Low volatility = Lower risk"
    },
    "TREND": {
        "name": "Price Trend",
        "description": "The general direction of price movement over time.",
        "interpretation": "UPTREND = Rising prices, DOWNTREND = Falling prices, SIDEWAYS = Range-bound"
    }
}

def get_term_explanation(term: str) -> dict:
    """Get explanation for a financial term"""
    return FINANCIAL_TERMS.get(term.upper(), {
        "name": term,
        "description": "Term not found in glossary",
        "interpretation": "No interpretation available"
    })