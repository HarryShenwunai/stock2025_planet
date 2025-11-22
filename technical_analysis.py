from typing import List, Tuple, Dict, Optional
import numpy as np

class TechnicalAnalyzer:
    """
    Advanced technical analysis engine with scientifically accurate indicators.
    
    Implements industry-standard technical indicators following academic research
    and professional trading methodologies.
    """
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """
        Calculate Exponential Moving Average using proper smoothing factor.
        
        EMA = Price(t) * k + EMA(t-1) * (1-k)
        where k = 2 / (period + 1)
        """
        if len(prices) < period:
            return sum(prices) / len(prices)
        
        k = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA
        
        for price in prices[period:]:
            ema = price * k + ema * (1 - k)
        
        return ema
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """
        Calculate Relative Strength Index using Wilder's smoothing method.
        
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        
        Uses exponential moving average for smoothing as per J. Welles Wilder Jr.
        """
        if len(prices) < period + 1:
            return 50.0
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages (SMA for first period)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Apply Wilder's smoothing for subsequent periods
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            # When there are no losses, RSI approaches 100
            # Cap at 99.9 to avoid extreme values that may indicate data issues
            return 99.9
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Cap RSI at reasonable bounds (0.1 to 99.9)
        rsi = max(0.1, min(99.9, rsi))
        
        return round(rsi, 2)
    
    @staticmethod
    def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line
        
        Developed by Gerald Appel in the late 1970s.
        """
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        # Calculate EMAs
        ema_fast = TechnicalAnalyzer.calculate_ema(prices, fast)
        ema_slow = TechnicalAnalyzer.calculate_ema(prices, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # For signal line, we need MACD history
        # Simplified: calculate signal from recent prices
        macd_history = []
        for i in range(slow, len(prices)):
            ema_f = TechnicalAnalyzer.calculate_ema(prices[:i+1], fast)
            ema_s = TechnicalAnalyzer.calculate_ema(prices[:i+1], slow)
            macd_history.append(ema_f - ema_s)
        
        if len(macd_history) < signal:
            signal_line = macd_line
        else:
            signal_line = TechnicalAnalyzer.calculate_ema(macd_history, signal)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return round(macd_line, 4), round(signal_line, 4), round(histogram, 4)
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """
        Calculate Bollinger Bands.
        
        Middle Band = SMA(20)
        Upper Band = Middle Band + (Standard Deviation * 2)
        Lower Band = Middle Band - (Standard Deviation * 2)
        
        Developed by John Bollinger in the 1980s.
        """
        if len(prices) < period:
            sma = np.mean(prices)
            std = np.std(prices)
        else:
            recent_prices = prices[-period:]
            sma = np.mean(recent_prices)
            std = np.std(recent_prices)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return round(upper_band, 2), round(sma, 2), round(lower_band, 2)
    
    @staticmethod
    def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[float, float]:
        """
        Calculate Stochastic Oscillator (%K and %D).
        
        %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        %D = 3-period SMA of %K
        
        Developed by George Lane in the late 1950s.
        """
        if len(closes) < period:
            return 50.0, 50.0
        
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = 100 * (current_close - lowest_low) / (highest_high - lowest_low)
        
        # Calculate %D (simplified as we need %K history)
        k_values = []
        for i in range(max(0, len(closes) - period - 2), len(closes)):
            h = highs[max(0, i-period+1):i+1]
            l = lows[max(0, i-period+1):i+1]
            c = closes[i]
            if len(h) > 0 and len(l) > 0:
                hh = max(h)
                ll = min(l)
                if hh != ll:
                    k_values.append(100 * (c - ll) / (hh - ll))
        
        d_percent = np.mean(k_values[-3:]) if len(k_values) >= 3 else k_percent
        
        return round(k_percent, 2), round(d_percent, 2)
    
    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).
        
        True Range = max(High - Low, abs(High - Previous Close), abs(Low - Previous Close))
        ATR = EMA of True Range
        
        Developed by J. Welles Wilder Jr.
        """
        if len(closes) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(closes)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i-1]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        if len(true_ranges) < period:
            return np.mean(true_ranges)
        
        # Use Wilder's smoothing
        atr = np.mean(true_ranges[:period])
        for tr in true_ranges[period:]:
            atr = (atr * (period - 1) + tr) / period
        
        return round(atr, 2)
    
    @staticmethod
    def calculate_obv(closes: List[float], volumes: List[int]) -> float:
        """
        Calculate On-Balance Volume (OBV).
        
        OBV tracks cumulative volume based on price direction.
        Developed by Joseph Granville.
        """
        if len(closes) < 2 or len(volumes) < 2:
            return 0.0
        
        obv = 0
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
        
        return obv
    
    @staticmethod
    def find_peaks_and_troughs(prices: List[float], order: int = 3) -> Tuple[List[int], List[int]]:
        """
        Find local maxima (peaks) and minima (troughs) using scientific method.
        
        Uses the concept of order-n extrema: a point is a peak if it's the maximum
        within a window of ±order points.
        
        Returns:
            peaks: List of indices where peaks occur
            troughs: List of indices where troughs occur
        """
        if len(prices) < 2 * order + 1:
            return [], []
        
        peaks = []
        troughs = []
        
        for i in range(order, len(prices) - order):
            # Check if this is a peak (local maximum)
            window = prices[i-order:i+order+1]
            if prices[i] == max(window) and prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append(i)
            # Check if this is a trough (local minimum)
            elif prices[i] == min(window) and prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append(i)
        
        return peaks, troughs
    
    @staticmethod
    def calculate_linear_regression(prices: List[float]) -> Tuple[float, float, float]:
        """
        Calculate linear regression (trend line) using least squares method.
        
        Returns:
            slope: Rate of change (positive = uptrend, negative = downtrend)
            intercept: Y-intercept of the trend line
            r_squared: R² coefficient (goodness of fit, 0-1)
        """
        if len(prices) < 2:
            return 0.0, prices[0] if prices else 0.0, 0.0
        
        x = np.arange(len(prices))
        y = np.array(prices)
        
        # Calculate slope and intercept using least squares
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Calculate R² (coefficient of determination)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return slope, intercept, r_squared
    
    @staticmethod
    def detect_support_resistance(prices: List[float], threshold: float = 0.02) -> Tuple[List[float], List[float]]:
        """
        Detect support and resistance levels using clustering algorithm.
        
        Uses density-based clustering to find price levels where the price
        has repeatedly tested and bounced.
        
        Args:
            threshold: Price similarity threshold (2% by default)
        
        Returns:
            support_levels: List of support price levels
            resistance_levels: List of resistance price levels
        """
        if len(prices) < 5:
            return [], []
        
        peaks, troughs = TechnicalAnalyzer.find_peaks_and_troughs(prices)
        
        if not peaks and not troughs:
            return [], []
        
        # Cluster peaks (resistance) and troughs (support)
        def cluster_levels(indices, prices, threshold):
            if not indices:
                return []
            
            levels = [prices[i] for i in indices]
            levels.sort()
            
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] <= threshold:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            if current_cluster:
                clusters.append(np.mean(current_cluster))
            
            # Filter clusters with at least 2 touches
            return [c for c in clusters if len([l for l in levels if abs(l - c) / c <= threshold]) >= 2]
        
        support_levels = cluster_levels(troughs, prices, threshold)
        resistance_levels = cluster_levels(peaks, prices, threshold)
        
        return support_levels, resistance_levels
    
    @staticmethod
    def detect_pattern(prices: List[float]) -> Dict:
        """
        Scientifically detect chart patterns using statistical methods.
        
        Combines multiple analytical approaches:
        1. Linear regression for trend analysis
        2. Peak/trough detection for pattern recognition
        3. Statistical measures (volatility, correlation)
        4. Geometric pattern matching
        
        Returns:
            Dictionary with pattern name, confidence, and characteristics
        """
        if len(prices) < 5:
            return {
                "pattern": "insufficient_data",
                "confidence": 0.0,
                "description": "Not enough data points for pattern analysis"
            }
        
        # 1. Linear Regression Analysis
        slope, intercept, r_squared = TechnicalAnalyzer.calculate_linear_regression(prices)
        
        # Normalize slope to percentage change per period
        price_mean = np.mean(prices)
        slope_pct = (slope / price_mean * 100) if price_mean > 0 else 0
        
        # 2. Find peaks and troughs
        peaks, troughs = TechnicalAnalyzer.find_peaks_and_troughs(prices)
        
        # 3. Calculate statistical measures
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0
        skewness = TechnicalAnalyzer._calculate_skewness(returns) if len(returns) > 2 else 0
        
        # 4. Pattern Detection Logic
        
        # Strong Trend Detection (R² > 0.7 indicates strong linear trend)
        if r_squared > 0.7:
            if slope_pct > 0.5:
                return {
                    "pattern": "strong_uptrend",
                    "confidence": r_squared,
                    "slope": round(slope_pct, 2),
                    "description": f"Strong upward trend with {round(slope_pct, 2)}% daily increase",
                    "r_squared": round(r_squared, 3)
                }
            elif slope_pct < -0.5:
                return {
                    "pattern": "strong_downtrend",
                    "confidence": r_squared,
                    "slope": round(slope_pct, 2),
                    "description": f"Strong downward trend with {round(slope_pct, 2)}% daily decrease",
                    "r_squared": round(r_squared, 3)
                }
        
        # Head and Shoulders Pattern (3 peaks with middle highest)
        if len(peaks) >= 3 and len(prices) >= 10:
            recent_peaks = sorted([(i, prices[i]) for i in peaks[-3:]])
            if len(recent_peaks) == 3:
                left, head, right = recent_peaks
                # Check if middle peak is highest and shoulders are similar
                if (head[1] > left[1] and head[1] > right[1] and
                    abs(left[1] - right[1]) / left[1] < 0.05):
                    confidence = 1 - abs(left[1] - right[1]) / left[1]
                    return {
                        "pattern": "head_and_shoulders",
                        "confidence": round(confidence, 2),
                        "description": "Classic reversal pattern - potential bearish signal",
                        "neckline": round(min(left[1], right[1]), 2)
                    }
        
        # Inverse Head and Shoulders (3 troughs with middle lowest)
        if len(troughs) >= 3 and len(prices) >= 10:
            recent_troughs = sorted([(i, prices[i]) for i in troughs[-3:]])
            if len(recent_troughs) == 3:
                left, head, right = recent_troughs
                if (head[1] < left[1] and head[1] < right[1] and
                    abs(left[1] - right[1]) / left[1] < 0.05):
                    confidence = 1 - abs(left[1] - right[1]) / left[1]
                    return {
                        "pattern": "inverse_head_and_shoulders",
                        "confidence": round(confidence, 2),
                        "description": "Classic reversal pattern - potential bullish signal",
                        "neckline": round(max(left[1], right[1]), 2)
                    }
        
        # Double Top Pattern (2 similar peaks near end)
        if len(peaks) >= 2:
            last_two_peaks = [(i, prices[i]) for i in peaks[-2:]]
            if len(last_two_peaks) == 2:
                peak1, peak2 = last_two_peaks
                price_diff = abs(peak1[1] - peak2[1]) / peak1[1]
                if price_diff < 0.03:  # Within 3%
                    confidence = 1 - price_diff
                    return {
                        "pattern": "double_top",
                        "confidence": round(confidence, 2),
                        "description": "Bearish reversal pattern - resistance level tested twice",
                        "resistance_level": round(np.mean([peak1[1], peak2[1]]), 2)
                    }
        
        # Double Bottom Pattern (2 similar troughs near end)
        if len(troughs) >= 2:
            last_two_troughs = [(i, prices[i]) for i in troughs[-2:]]
            if len(last_two_troughs) == 2:
                trough1, trough2 = last_two_troughs
                price_diff = abs(trough1[1] - trough2[1]) / trough1[1]
                if price_diff < 0.03:  # Within 3%
                    confidence = 1 - price_diff
                    return {
                        "pattern": "double_bottom",
                        "confidence": round(confidence, 2),
                        "description": "Bullish reversal pattern - support level tested twice",
                        "support_level": round(np.mean([trough1[1], trough2[1]]), 2)
                    }
        
        # Triangle Pattern (converging highs and lows)
        if len(peaks) >= 2 and len(troughs) >= 2:
            peak_slope = (prices[peaks[-1]] - prices[peaks[0]]) / len(peaks)
            trough_slope = (prices[troughs[-1]] - prices[troughs[0]]) / len(troughs)
            
            # Symmetrical triangle (both converging)
            if peak_slope < 0 and trough_slope > 0:
                convergence = abs(peak_slope) + abs(trough_slope)
                return {
                    "pattern": "symmetrical_triangle",
                    "confidence": min(convergence / price_mean * 100, 1.0),
                    "description": "Consolidation pattern - potential breakout imminent",
                    "direction": "neutral"
                }
            # Ascending triangle
            elif abs(peak_slope) < price_mean * 0.01 and trough_slope > 0:
                return {
                    "pattern": "ascending_triangle",
                    "confidence": 0.7,
                    "description": "Bullish continuation pattern",
                    "direction": "bullish"
                }
            # Descending triangle
            elif peak_slope < 0 and abs(trough_slope) < price_mean * 0.01:
                return {
                    "pattern": "descending_triangle",
                    "confidence": 0.7,
                    "description": "Bearish continuation pattern",
                    "direction": "bearish"
                }
        
        # Channel Detection (parallel support and resistance)
        support_levels, resistance_levels = TechnicalAnalyzer.detect_support_resistance(prices)
        if len(support_levels) >= 1 and len(resistance_levels) >= 1:
            channel_width = resistance_levels[0] - support_levels[0]
            if abs(channel_width / price_mean) > 0.02:  # At least 2% channel
                return {
                    "pattern": "trading_channel",
                    "confidence": 0.75,
                    "description": "Price trading in defined channel",
                    "support": round(support_levels[0], 2),
                    "resistance": round(resistance_levels[0], 2),
                    "channel_width_pct": round(channel_width / price_mean * 100, 2)
                }
        
        # Consolidation (low volatility, no clear trend)
        if volatility < 0.015 and abs(slope_pct) < 0.3:
            return {
                "pattern": "consolidation",
                "confidence": round(1 - volatility / 0.015, 2),
                "description": "Low volatility consolidation - potential breakout setup",
                "volatility": round(volatility * 100, 2)
            }
        
        # High Volatility / Choppy Market
        if volatility > 0.04:
            return {
                "pattern": "high_volatility",
                "confidence": round(min(volatility / 0.04, 1.0), 2),
                "description": "High volatility - difficult to predict direction",
                "volatility": round(volatility * 100, 2)
            }
        
        # Default: Sideways/Ranging
        return {
            "pattern": "sideways",
            "confidence": 0.5,
            "description": "No clear pattern detected - ranging market",
            "slope": round(slope_pct, 2),
            "volatility": round(volatility * 100, 2)
        }
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness (measure of asymmetry in distribution)"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        n = len(data)
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3)
        return skew
    
    @staticmethod
    def calculate_hurst_exponent(prices: List[float]) -> float:
        """
        Calculate Hurst Exponent (H) to measure the long-term memory of time series.
        
        H < 0.5: Mean-reverting (anti-persistent) - oversold/overbought tend to revert
        H = 0.5: Random walk (Geometric Brownian Motion) - unpredictable
        H > 0.5: Trending (persistent) - trends tend to continue
        
        Uses rescaled range (R/S) analysis developed by H.E. Hurst.
        """
        if len(prices) < 20:
            return 0.5  # Assume random walk for insufficient data
        
        prices = np.array(prices)
        lags = range(2, min(20, len(prices) // 2))
        
        # Calculate R/S for different time lags
        tau = []
        rs = []
        
        for lag in lags:
            # Divide series into subseries
            subseries_count = len(prices) // lag
            rs_values = []
            
            for i in range(subseries_count):
                subseries = prices[i*lag:(i+1)*lag]
                
                # Mean
                mean = np.mean(subseries)
                
                # Mean-adjusted series
                Y = subseries - mean
                
                # Cumulative deviate series
                Z = np.cumsum(Y)
                
                # Range
                R = np.max(Z) - np.min(Z)
                
                # Standard deviation
                S = np.std(subseries)
                
                if S != 0:
                    rs_values.append(R / S)
            
            if rs_values:
                tau.append(lag)
                rs.append(np.mean(rs_values))
        
        if len(tau) < 2:
            return 0.5
        
        # Linear regression on log-log plot
        log_tau = np.log(tau)
        log_rs = np.log(rs)
        
        # Calculate Hurst exponent (slope)
        slope, _, _ = TechnicalAnalyzer.calculate_linear_regression(log_rs)
        
        return round(min(max(slope, 0), 1), 3)  # Clamp between 0 and 1
    
    @staticmethod
    def calculate_fractal_dimension(prices: List[float]) -> float:
        """
        Calculate Fractal Dimension using Higuchi's method.
        
        Measures the complexity and self-similarity of the time series.
        D closer to 1: Smooth, trending
        D closer to 2: Rough, noisy, random
        
        Developed by Higuchi in 1988.
        """
        if len(prices) < 10:
            return 1.5
        
        prices = np.array(prices)
        N = len(prices)
        
        k_max = min(10, N // 4)
        k_values = range(1, k_max + 1)
        Lk = []
        
        for k in k_values:
            Lm = []
            for m in range(k):
                # Construct subsequence
                indices = np.arange(m, N, k)
                if len(indices) < 2:
                    continue
                
                # Calculate length
                subseries = prices[indices]
                length = np.sum(np.abs(np.diff(subseries))) * (N - 1) / (len(indices) * k)
                Lm.append(length)
            
            if Lm:
                Lk.append(np.mean(Lm))
        
        if len(Lk) < 2:
            return 1.5
        
        # Linear regression on log-log plot
        log_k = np.log(k_values[:len(Lk)])
        log_Lk = np.log(Lk)
        
        slope, _, _ = TechnicalAnalyzer.calculate_linear_regression(log_Lk)
        
        # Fractal dimension is negative of slope
        fractal_dim = -slope
        
        return round(min(max(fractal_dim, 1), 2), 3)
    
    @staticmethod
    def perform_fourier_analysis(prices: List[float], top_n: int = 3) -> Dict:
        """
        Perform Fast Fourier Transform (FFT) to identify dominant cycles.
        
        Identifies periodicities in price movement using frequency domain analysis.
        Useful for detecting cyclical patterns and seasonality.
        """
        if len(prices) < 10:
            return {"dominant_periods": [], "power_spectrum": []}
        
        # Detrend the data
        slope, intercept, _ = TechnicalAnalyzer.calculate_linear_regression(prices)
        x = np.arange(len(prices))
        trend = slope * x + intercept
        detrended = np.array(prices) - trend
        
        # Apply FFT
        fft = np.fft.fft(detrended)
        power = np.abs(fft) ** 2
        freqs = np.fft.fftfreq(len(detrended))
        
        # Only consider positive frequencies
        positive_freqs_mask = freqs > 0
        freqs = freqs[positive_freqs_mask]
        power = power[positive_freqs_mask]
        
        # Find dominant frequencies
        sorted_indices = np.argsort(power)[::-1]
        
        dominant_periods = []
        for i in range(min(top_n, len(sorted_indices))):
            idx = sorted_indices[i]
            if freqs[idx] > 0:
                period = 1 / freqs[idx]
                if 2 <= period <= len(prices) / 2:  # Reasonable periods only
                    dominant_periods.append({
                        "period_days": round(period, 1),
                        "power": round(float(power[idx]), 2),
                        "frequency": round(float(freqs[idx]), 4)
                    })
        
        return {
            "dominant_periods": dominant_periods,
            "interpretation": "Cyclical pattern detected" if dominant_periods else "No clear cycles"
        }
    
    @staticmethod
    def calculate_entropy(prices: List[float], bins: int = 10) -> float:
        """
        Calculate Shannon Entropy to measure unpredictability/randomness.
        
        High entropy: High unpredictability, random behavior
        Low entropy: Low unpredictability, more predictable patterns
        
        Based on information theory by Claude Shannon.
        """
        if len(prices) < bins:
            return 0.0
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Create histogram
        hist, _ = np.histogram(returns, bins=bins)
        
        # Calculate probabilities
        probabilities = hist / len(returns)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return round(entropy, 3)
    
    @staticmethod
    def calculate_sharpe_ratio(prices: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio - risk-adjusted return measure.
        
        Sharpe Ratio = (Mean Return - Risk Free Rate) / Standard Deviation of Returns
        
        Higher is better. Developed by William F. Sharpe.
        > 1: Good
        > 2: Very Good
        > 3: Excellent
        """
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        
        mean_return = np.mean(returns) * 252  # Annualized (assuming daily data)
        std_return = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        if std_return == 0:
            return 0.0
        
        sharpe = (mean_return - risk_free_rate) / std_return
        
        return round(sharpe, 3)
    
    @staticmethod
    def calculate_sortino_ratio(prices: List[float], risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino Ratio - downside risk-adjusted return.
        
        Like Sharpe but only considers downside volatility (negative returns).
        Developed by Frank A. Sortino.
        """
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        
        mean_return = np.mean(returns) * 252  # Annualized
        
        # Downside deviation (only negative returns)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')  # Perfect - no downside
        
        downside_std = np.std(negative_returns) * np.sqrt(252)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return - risk_free_rate) / downside_std
        
        return round(sortino, 3)
    
    @staticmethod
    def calculate_maximum_drawdown(prices: List[float]) -> Dict:
        """
        Calculate Maximum Drawdown (MDD) - largest peak-to-trough decline.
        
        Critical risk metric for understanding worst-case scenarios.
        """
        if len(prices) < 2:
            return {"max_drawdown_pct": 0.0, "peak_price": 0.0, "trough_price": 0.0}
        
        prices = np.array(prices)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)
        
        # Calculate drawdown at each point
        drawdown = (prices - running_max) / running_max
        
        # Find maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        # Find the peak before this drawdown
        peak_idx = np.argmax(prices[:max_dd_idx+1]) if max_dd_idx > 0 else 0
        
        return {
            "max_drawdown_pct": round(max_dd * 100, 2),
            "peak_price": round(float(prices[peak_idx]), 2),
            "trough_price": round(float(prices[max_dd_idx]), 2),
            "recovery_needed_pct": round(abs(max_dd / (1 + max_dd) * 100), 2) if max_dd < 0 else 0
        }
    
    @staticmethod
    def calculate_correlation_dimension(prices: List[float]) -> float:
        """
        Calculate Correlation Dimension to measure the complexity of attractors.
        
        Estimates the fractal dimension of the system's attractor.
        Used in chaos theory to analyze deterministic systems.
        """
        if len(prices) < 20:
            return 1.0
        
        prices = np.array(prices)
        
        # Normalize prices
        normalized = (prices - np.mean(prices)) / np.std(prices)
        
        # Calculate distances between all pairs
        distances = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                dist = abs(normalized[i] - normalized[j])
                distances.append(dist)
        
        if not distances:
            return 1.0
        
        distances = np.array(distances)
        
        # Count pairs within different radii
        radii = np.percentile(distances, [10, 20, 30, 40, 50])
        counts = [np.sum(distances <= r) for r in radii]
        
        # Log-log regression
        log_radii = np.log(radii + 1e-10)
        log_counts = np.log(np.array(counts) + 1)
        
        slope, _, _ = TechnicalAnalyzer.calculate_linear_regression(log_counts)
        
        return round(min(max(slope, 0.5), 3.0), 3)
    
    @staticmethod
    def comprehensive_analysis(prices: List[float], highs: Optional[List[float]] = None, 
                              lows: Optional[List[float]] = None, volumes: Optional[List[int]] = None) -> Dict:
        """
        Perform comprehensive scientific technical analysis.
        
        Combines classical technical indicators with advanced statistical methods:
        - Traditional indicators (RSI, MACD, Bollinger Bands, etc.)
        - Time series analysis (Hurst, Fractal Dimension, Fourier)
        - Risk metrics (Sharpe, Sortino, Maximum Drawdown)
        - Information theory (Entropy)
        - Pattern recognition (Statistical and geometric)
        
        Returns a dictionary with all indicators and their interpretations.
        """
        if highs is None:
            highs = prices
        if lows is None:
            lows = prices
        if volumes is None:
            volumes = [1000000] * len(prices)
        
        # Traditional Technical Indicators
        rsi = TechnicalAnalyzer.calculate_rsi(prices)
        macd, signal, histogram = TechnicalAnalyzer.calculate_macd(prices)
        upper_bb, middle_bb, lower_bb = TechnicalAnalyzer.calculate_bollinger_bands(prices)
        stoch_k, stoch_d = TechnicalAnalyzer.calculate_stochastic(highs, lows, prices)
        atr = TechnicalAnalyzer.calculate_atr(highs, lows, prices)
        obv = TechnicalAnalyzer.calculate_obv(prices, volumes)
        pattern = TechnicalAnalyzer.detect_pattern(prices)
        
        # Advanced Statistical Analysis
        hurst = TechnicalAnalyzer.calculate_hurst_exponent(prices)
        fractal_dim = TechnicalAnalyzer.calculate_fractal_dimension(prices)
        fourier = TechnicalAnalyzer.perform_fourier_analysis(prices)
        entropy = TechnicalAnalyzer.calculate_entropy(prices)
        
        # Risk Metrics
        sharpe = TechnicalAnalyzer.calculate_sharpe_ratio(prices)
        sortino = TechnicalAnalyzer.calculate_sortino_ratio(prices)
        max_dd = TechnicalAnalyzer.calculate_maximum_drawdown(prices)
        
        # Linear Regression
        slope, intercept, r_squared = TechnicalAnalyzer.calculate_linear_regression(prices)
        
        # Support & Resistance
        support_levels, resistance_levels = TechnicalAnalyzer.detect_support_resistance(prices)
        
        # Interpretations
        rsi_signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        macd_signal = "bullish" if histogram > 0 else "bearish"
        bb_position = "near_upper" if prices[-1] > middle_bb + (upper_bb - middle_bb) * 0.5 else \
                     "near_lower" if prices[-1] < middle_bb - (middle_bb - lower_bb) * 0.5 else "middle"
        stoch_signal = "oversold" if stoch_k < 20 else "overbought" if stoch_k > 80 else "neutral"
        
        hurst_interpretation = (
            "mean_reverting" if hurst < 0.45 else
            "random_walk" if 0.45 <= hurst <= 0.55 else
            "trending"
        )
        
        market_quality = (
            "excellent" if sharpe > 2 else
            "good" if sharpe > 1 else
            "fair" if sharpe > 0 else
            "poor"
        )
        
        return {
            "classical_indicators": {
                "rsi": {"value": rsi, "signal": rsi_signal},
                "macd": {
                    "macd_line": macd,
                    "signal_line": signal,
                    "histogram": histogram,
                    "signal": macd_signal
                },
                "bollinger_bands": {
                    "upper": upper_bb,
                    "middle": middle_bb,
                    "lower": lower_bb,
                    "position": bb_position
                },
                "stochastic": {
                    "k": stoch_k,
                    "d": stoch_d,
                    "signal": stoch_signal
                },
                "atr": atr,
                "obv": obv
            },
            "pattern_recognition": pattern,
            "statistical_analysis": {
                "hurst_exponent": {
                    "value": hurst,
                    "interpretation": hurst_interpretation,
                    "description": "Measures market memory and predictability"
                },
                "fractal_dimension": {
                    "value": fractal_dim,
                    "interpretation": "smooth" if fractal_dim < 1.5 else "rough",
                    "description": "Measures complexity of price movement"
                },
                "entropy": {
                    "value": entropy,
                    "interpretation": "high_randomness" if entropy > 2.5 else "low_randomness",
                    "description": "Measures unpredictability"
                },
                "linear_regression": {
                    "slope": round(slope, 4),
                    "r_squared": round(r_squared, 3),
                    "trend_strength": "strong" if r_squared > 0.7 else "weak"
                }
            },
            "cyclical_analysis": fourier,
            "risk_metrics": {
                "sharpe_ratio": {
                    "value": sharpe,
                    "quality": market_quality,
                    "description": "Risk-adjusted return measure"
                },
                "sortino_ratio": {
                    "value": sortino,
                    "description": "Downside risk-adjusted return"
                },
                "maximum_drawdown": max_dd
            },
            "support_resistance": {
                "support_levels": [round(s, 2) for s in support_levels[:3]],
                "resistance_levels": [round(r, 2) for r in resistance_levels[:3]]
            },
            "current_price": prices[-1],
            "price_statistics": {
                "5d_change_pct": round(((prices[-1] - prices[-5]) / prices[-5] * 100), 2) if len(prices) >= 5 else 0,
                "20d_volatility": round(np.std(prices[-20:]) / np.mean(prices[-20:]) * 100, 2) if len(prices) >= 20 else 0,
                "trend_direction": "up" if slope > 0 else "down"
            }
        }
