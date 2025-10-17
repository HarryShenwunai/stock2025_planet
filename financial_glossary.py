FINANCIAL_TERMS = {
    "RSI": {
        "name": "Relative Strength Index",
        "description": "A momentum oscillator developed by J. Welles Wilder Jr. that measures the speed and magnitude of price changes. Ranges from 0-100.",
        "interpretation": "Above 70 = Overbought (potential sell signal), Below 30 = Oversold (potential buy signal), 50 = Neutral",
        "formula": "RSI = 100 - (100 / (1 + RS)) where RS = Average Gain / Average Loss",
        "period": "Typically 14 periods"
    },
    "MACD": {
        "name": "Moving Average Convergence Divergence",
        "description": "A trend-following momentum indicator developed by Gerald Appel that shows the relationship between two exponential moving averages (EMA).",
        "interpretation": "MACD above Signal Line = Bullish, MACD below Signal Line = Bearish. Positive histogram = Upward momentum, Negative = Downward momentum",
        "formula": "MACD Line = EMA(12) - EMA(26), Signal Line = EMA(9) of MACD Line, Histogram = MACD - Signal",
        "components": "MACD Line, Signal Line, Histogram"
    },
    "BOLLINGER_BANDS": {
        "name": "Bollinger Bands",
        "description": "A volatility indicator developed by John Bollinger consisting of a middle band (SMA) and two outer bands at standard deviations.",
        "interpretation": "Price near upper band = Overbought conditions, Price near lower band = Oversold conditions. Band squeeze = Low volatility (potential breakout)",
        "formula": "Middle Band = SMA(20), Upper Band = SMA(20) + (2 × Standard Deviation), Lower Band = SMA(20) - (2 × Standard Deviation)",
        "use_case": "Identifies overbought/oversold conditions and volatility"
    },
    "STOCHASTIC": {
        "name": "Stochastic Oscillator",
        "description": "A momentum indicator developed by George Lane that compares a security's closing price to its price range over a specific period.",
        "interpretation": "Above 80 = Overbought, Below 20 = Oversold. %K crossing above %D = Buy signal, %K crossing below %D = Sell signal",
        "formula": "%K = 100 × (Close - Lowest Low) / (Highest High - Lowest Low), %D = 3-period SMA of %K",
        "period": "Typically 14 periods"
    },
    "ATR": {
        "name": "Average True Range",
        "description": "A volatility indicator developed by J. Welles Wilder Jr. that measures market volatility by decomposing the entire range of price movement.",
        "interpretation": "High ATR = High volatility (wider stops needed), Low ATR = Low volatility (tighter stops possible)",
        "formula": "TR = max(High - Low, abs(High - Previous Close), abs(Low - Previous Close)), ATR = EMA of TR",
        "use_case": "Setting stop-loss levels and position sizing"
    },
    "OBV": {
        "name": "On-Balance Volume",
        "description": "A cumulative volume-based indicator developed by Joseph Granville that relates volume to price change.",
        "interpretation": "Rising OBV = Accumulation (bullish), Falling OBV = Distribution (bearish). OBV divergence from price can signal reversals",
        "formula": "If Close > Previous Close: OBV = Previous OBV + Volume, If Close < Previous Close: OBV = Previous OBV - Volume",
        "use_case": "Confirms price trends and predicts breakouts"
    },
    "EMA": {
        "name": "Exponential Moving Average",
        "description": "A type of moving average that gives more weight to recent prices, making it more responsive to new information.",
        "interpretation": "Price above EMA = Bullish trend, Price below EMA = Bearish trend. EMA crossovers signal trend changes",
        "formula": "EMA = Price(t) × k + EMA(t-1) × (1-k) where k = 2 / (period + 1)",
        "common_periods": "12, 26, 50, 200 periods"
    },
    "VOLATILITY": {
        "name": "Price Volatility",
        "description": "A statistical measure of the dispersion of returns, representing the degree of variation in prices over time.",
        "interpretation": "High volatility = Higher risk and potential returns, Low volatility = Lower risk and steadier prices",
        "formula": "Standard Deviation of price returns over a specific period",
        "use_case": "Risk assessment and option pricing"
    },
    "TREND": {
        "name": "Price Trend",
        "description": "The general direction of price movement over time, fundamental to technical analysis.",
        "interpretation": "UPTREND = Series of higher highs and higher lows, DOWNTREND = Series of lower highs and lower lows, SIDEWAYS = Range-bound movement",
        "types": "Primary (long-term), Secondary (medium-term), Minor (short-term)",
        "principle": "The trend is your friend - trade in the direction of the trend"
    },
    "SUPPORT": {
        "name": "Support Level",
        "description": "A price level where buying pressure is strong enough to prevent the price from declining further.",
        "interpretation": "Price bouncing off support = Bullish signal, Break below support = Bearish signal (becomes resistance)",
        "use_case": "Identifying entry points and setting stop-loss levels"
    },
    "RESISTANCE": {
        "name": "Resistance Level",
        "description": "A price level where selling pressure is strong enough to prevent the price from rising further.",
        "interpretation": "Price rejected at resistance = Bearish signal, Break above resistance = Bullish signal (becomes support)",
        "use_case": "Identifying exit points and target prices"
    },
    "DIVERGENCE": {
        "name": "Indicator Divergence",
        "description": "When price and an indicator move in opposite directions, potentially signaling a trend reversal.",
        "interpretation": "Bullish divergence = Price makes lower low, indicator makes higher low. Bearish divergence = Price makes higher high, indicator makes lower high",
        "reliability": "Stronger on longer timeframes and with multiple indicator confirmation"
    },
    "VOLUME": {
        "name": "Trading Volume",
        "description": "The number of shares or contracts traded in a security or market during a given period.",
        "interpretation": "High volume on price increase = Strong bullish signal, High volume on price decrease = Strong bearish signal. Low volume = Weak conviction",
        "principle": "Volume precedes price - changes in volume often precede price movements"
    },
    "MOMENTUM": {
        "name": "Price Momentum",
        "description": "The rate of acceleration of a security's price or volume, measuring the speed of price changes.",
        "interpretation": "Increasing momentum = Trend acceleration, Decreasing momentum = Potential trend weakening or reversal",
        "indicators": "RSI, MACD, Stochastic, ROC (Rate of Change)"
    }
}

def get_term_explanation(term: str) -> dict:
    """Get explanation for a financial term"""
    return FINANCIAL_TERMS.get(term.upper(), {
        "name": term,
        "description": "Term not found in glossary",
        "interpretation": "No interpretation available"
    })

def get_all_terms() -> dict:
    """Get all financial terms in the glossary"""
    return FINANCIAL_TERMS

def search_terms(keyword: str) -> dict:
    """Search for terms containing the keyword"""
    keyword = keyword.lower()
    results = {}
    for key, value in FINANCIAL_TERMS.items():
        if (keyword in key.lower() or 
            keyword in value.get("name", "").lower() or 
            keyword in value.get("description", "").lower()):
            results[key] = value
    return results
