"""
Combined Financial Analysis Web Application
==========================================

This application combines three different examples into a unified web interface:
- Flask-based events and stock data display
- Advanced AI Financial Agent with comprehensive analysis
- Simple FastAPI endpoints for stock and news data

Features:
- Multi-framework support (Flask + FastAPI)
- Real-time stock data and analysis
- News sentiment analysis
- Technical indicators (RSI, MACD)
- Interactive charts and visualizations
- RESTful API endpoints
- Modern responsive UI

Author: Combined Examples v1.0
Date: October 2025
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
# Removed Flask import - using FastAPI only for better compatibility
from textblob import TextBlob

from data_structures import MarketData, NewsData, AnalysisResult, DataCache
from technical_analysis import TechnicalAnalyzer
from api_client import get_api_config, fetch_api_data, fetch_stock_data, fetch_alpha_vantage_stock, fetch_news_data
from sentiment_analysis import SentimentAnalyzer
from financial_agent import CombinedFinancialAgent
# Removed template imports - API only


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===== GLOBAL AGENT =====
agent = None

def init_agent():
    """Initialize the combined agent"""
    global agent
    try:
        logger.info("Starting agent initialization...")
        agent = CombinedFinancialAgent()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        logger.exception("Full traceback:")
        agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application starting up...")
    logger.info("Startup completed - ready to serve requests")
    yield
    # Shutdown
    logger.info("Application shutting down...")

# ===== FASTAPI APPLICATION =====
app = FastAPI(
    title="Combined Financial Analysis API",
    description="Unified financial analysis platform combining multiple data sources and AI analysis",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ===== API ENDPOINTS =====
@app.get("/", response_class=PlainTextResponse)
async def root():
    """Root endpoint - minimal plain text for proxies"""
    logger.info("Root endpoint called - Railway health check")
    return "ok"

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return PlainTextResponse("internal error", status_code=500)

@app.get("/api/analyze/{symbol}")
async def analyze_symbol(symbol: str):
    """AI analysis endpoint (from example2.py)"""
    global agent
    if agent is None:
        logger.info("Initializing agent on first request...")
        init_agent()
        if agent is None:
            raise HTTPException(status_code=503, detail="Service unavailable: Agent initialization failed")
    try:
        result = await agent.perform_comprehensive_analysis(symbol.upper())
        return asdict(result)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{symbol}")
async def get_analysis_history(symbol: str):
    """Get analysis history (from example2.py)"""
    try:
        global agent
        if agent is None:
            logger.info("Initializing agent for history request...")
            init_agent()
        
        if agent is None or agent.conn is None:
            return {"message": "Database not available", "history": []}
        
        cursor = agent.conn.cursor()
        cursor.execute(
            "SELECT * FROM analysis_history WHERE symbol = ? ORDER BY timestamp DESC LIMIT 10",
            (symbol.upper(),)
        )
        results = cursor.fetchall()
        
        history = []
        for row in results:
            history.append({
                'id': row[0],
                'symbol': row[1],
                'timestamp': row[2],
                'analysis': json.loads(row[3]),
                'confidence': row[4]
            })
        
        return history
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    logger.info("Health check called")
    return {"status": "ok"}

def _normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to [0, 1] range with clipping"""
    if max_val == min_val:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float with fallback"""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def _score_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a composite score with normalization and error handling.
    Returns: dict with score, confidence, and risk_level
    """
    tech = analysis.get("technical_indicators") or {}
    
    # 1. 优先使用已有的综合分数
    if "composite_score" in tech:
        try:
            comp_score = _safe_float(tech.get("composite_score"), 0.0)
            return {
                "score": round(comp_score, 3),
                "recommendation": analysis.get("recommendation", "UNKNOWN"),
                "confidence": _safe_float(analysis.get("confidence"), 0.0),
                "risk_level": analysis.get("risk_level", "UNKNOWN")
            }
        except Exception:
            pass
    
    # 2. 回退打分：提取因子并标准化
    try:
        # 基础建议分 (-2 到 +2)
        rec = (analysis.get("recommendation") or "").upper()
        rec_map = {"STRONG_BUY": 2.0, "BUY": 1.0, "HOLD": 0.0, "SELL": -1.0, "STRONG_SELL": -2.0}
        base = rec_map.get(rec, 0.0)
        
        # 提取原始值
        conf = _safe_float(analysis.get("confidence"), 0.0)
        r2 = _safe_float(tech.get("trend_r2"), 0.0)
        sharpe_raw = _safe_float(tech.get("sharpe"), 0.0)
        sortino_raw = _safe_float(tech.get("sortino"), 0.0)
        max_dd_raw = abs(_safe_float(tech.get("max_drawdown_pct"), 0.0))
        vol_raw = _safe_float(tech.get("volatility"), 0.0)
        
        # 标准化到 [0, 1] (避免量级差异)
        sharpe = _normalize(sharpe_raw, -2.0, 3.0)  # 通常 -2 到 3
        sortino = _normalize(sortino_raw, -2.0, 3.0)
        max_dd = _normalize(max_dd_raw, 0.0, 50.0)  # 0-50% 回撤
        vol = _normalize(vol_raw, 0.0, 100.0)  # 0-100% 波动率
        # r2 和 conf 已经是 [0, 1]
        
        # 数据完整度（用于置信度调整）
        fields_checked = ["trend_r2", "sharpe", "sortino", "max_drawdown_pct", "volatility"]
        available_fields = sum(1 for f in fields_checked if tech.get(f) is not None)
        data_completeness = available_fields / len(fields_checked)
        
        # 加权计算 (基础建议 + 技术因子)
        technical_score = (
            0.25 * r2 +           # 趋势强度
            0.20 * sharpe +       # 风险调整收益
            0.15 * sortino +      # 下行风险收益
            0.20 * (1 - max_dd) + # 回撤惩罚（反转）
            0.20 * (1 - vol)      # 波动率惩罚（反转）
        )
        
        # 综合分数：基础建议加权 + 技术分数
        # 基础建议范围 -2 到 +2，技术分数范围 0 到 1
        final_score = base * (0.7 + 0.3 * conf) + 2.0 * technical_score
        
        # 调整置信度
        adjusted_confidence = conf * data_completeness
        
        # 风险等级（基于波动率和回撤）
        risk_score = (max_dd_raw / 50.0 + vol_raw / 100.0) / 2
        if risk_score < 0.2:
            risk_level = "LOW"
        elif risk_score < 0.4:
            risk_level = "MODERATE"
        elif risk_score < 0.6:
            risk_level = "HIGH"
        else:
            risk_level = "VERY_HIGH"
        
        return {
            "score": round(final_score, 3),
            "recommendation": rec if rec else "UNKNOWN",
            "confidence": round(adjusted_confidence, 3),
            "risk_level": risk_level
        }
        
    except Exception as e:
        logger.warning(f"Score calculation error: {e}, using fallback")
        return {
            "score": 0.0,
            "recommendation": "UNKNOWN",
            "confidence": 0.0,
            "risk_level": "UNKNOWN"
        }

@app.get("/api/rank")
async def rank_analyzed(limit: int = 10):
    """
    Rank analyzed stocks using enriched multi-factor score.
    Returns sorted list with score, recommendation, confidence, and risk level.
    """
    try:
        if agent is None or agent.conn is None:
            return {"message": "Database not available", "ranked": []}
        cursor = agent.conn.cursor()
        cursor.execute(
            """
            SELECT t1.symbol, t1.timestamp, t1.analysis_result, t1.confidence
            FROM analysis_history t1
            JOIN (
                SELECT symbol, MAX(id) AS max_id
                FROM analysis_history
                GROUP BY symbol
            ) t2
            ON t1.symbol = t2.symbol AND t1.id = t2.max_id
            """
        )
        rows = cursor.fetchall()
        ranked: List[Dict[str, Any]] = []
        for symbol, ts, analysis_json, _c in rows:
            try:
                analysis = json.loads(analysis_json)
            except Exception:
                continue
            
            # _score_analysis 现在返回字典
            score_info = _score_analysis(analysis)
            
            ranked.append({
                "symbol": symbol,
                "score": score_info["score"],
                "recommendation": score_info["recommendation"],
                "confidence": score_info["confidence"],
                "risk_level": score_info["risk_level"],
                "current_price": analysis.get("current_price"),
                "timestamp": ts
            })
        
        # 按分数降序排序
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return {
            "count": len(ranked), 
            "ranked": ranked[: max(1, limit)],
            "note": "Score range: approximately -2 to +4 (higher is better)"
        }
    except Exception as e:
        logger.error(f"Rank error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{symbol}")
def get_stock_data_api(symbol: str):
    """Stock data endpoint (from example3.py)"""
    try:
        result = fetch_alpha_vantage_stock(symbol.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

@app.get("/api/news/{symbol}")
def get_news_data_api(symbol: str):
    """News data endpoint (from example3.py)"""
    try:
        result = fetch_news_data(symbol.upper())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news data: {str(e)}")

@app.get("/api/glossary")
async def get_glossary():
    """Get financial terms glossary"""
    from financial_glossary import FINANCIAL_TERMS
    return FINANCIAL_TERMS

@app.get("/api/glossary/{term}")
async def get_term_definition(term: str):
    """Get definition for a specific term"""
    from financial_glossary import get_term_explanation
    return get_term_explanation(term)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Financial Analysis API on port {port}")
    print("=" * 50)
    print("API Documentation: http://localhost:{port}/docs")
    print("ReDoc Documentation: http://localhost:{port}/redoc")
    print("Root Endpoint: http://localhost:{port}/")
    print("=" * 50)
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"Failed to start server: {e}")
        raise
