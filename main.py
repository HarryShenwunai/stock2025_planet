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
