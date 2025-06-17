from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
import sys
import os
from pathlib import Path

# Add the project root to the path so we can import modules correctly
sys.path.append(str(Path(__file__).parent.parent))

from backend.model_service import model_service
from backend.credibility_tracker import credibility_tracker
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backend/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fake News Detection API",
              description="API for detecting fake news and tracking source credibility")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsItem(BaseModel):
    """Request model for news text to analyze"""
    text: str
    url: Optional[HttpUrl] = None

class SourceQuery(BaseModel):
    """Request model for source credibility query"""
    domain: str

@app.get("/ping")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "API is running"}

@app.post("/predict")
async def predict_fake_news(news_item: NewsItem):
    """
    Predict if a news item is fake or real
    
    Args:
        news_item: The news text to analyze
        
    Returns:
        dict: A dictionary containing the prediction results
    """
    if not news_item.text or len(news_item.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text is too short. Please provide a longer text.")
    
    logger.info(f"Received prediction request: {news_item.text[:50]}...")
    
    result = model_service.get_prediction(news_item.text)
    
    if result["status"] == "error":
        logger.error(f"Prediction error: {result.get('error', 'Unknown error')}")
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to make prediction"))
    
    logger.info(f"Prediction result: {result['prediction']} (confidence: {result['confidence']:.4f})")
    
    response = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "fake_probability": result["fake_probability"],
        "real_probability": result["real_probability"]
    }
    
    # If URL is provided, extract metadata and update source credibility
    if news_item.url:
        try:
            # Extract metadata
            metadata = credibility_tracker.extract_metadata(str(news_item.url))
            
            # Record article and update source credibility
            source_info = credibility_tracker.record_article(
                str(news_item.url),
                metadata,
                result["prediction"],
                result["confidence"]
            )
            
            # Add source info to response
            response["source_info"] = source_info
            
        except Exception as e:
            logger.error(f"Error processing source credibility: {e}")
            # Don't fail the request if credibility tracking fails
    
    return response

@app.get("/source_score")
async def get_source_score(domain: str = Query(..., description="Domain to get credibility score for")):
    """
    Get the credibility score for a news source domain
    
    Args:
        domain: Domain to get credibility score for
        
    Returns:
        dict: Source information including credibility score
    """
    if not domain:
        raise HTTPException(status_code=400, detail="Domain parameter is required")
    
    logger.info(f"Received source score request for domain: {domain}")
    
    source_info = credibility_tracker.get_source_info(domain)
    
    return source_info

@app.get("/sources")
async def get_all_sources(
    limit: int = Query(100, description="Maximum number of sources to return"),
    sort_by: str = Query("credibility_score", description="Field to sort by"),
    ascending: bool = Query(False, description="Whether to sort in ascending order")
):
    """
    Get information about all tracked news sources
    
    Args:
        limit: Maximum number of sources to return
        sort_by: Field to sort by
        ascending: Whether to sort in ascending order
        
    Returns:
        list: List of source information dictionaries
    """
    logger.info(f"Received request for all sources (limit={limit}, sort_by={sort_by}, ascending={ascending})")
    
    sources = credibility_tracker.get_all_sources(limit=limit, sort_by=sort_by, ascending=ascending)
    
    return sources

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 