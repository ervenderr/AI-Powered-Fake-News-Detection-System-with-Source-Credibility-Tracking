from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model_service import model_service
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
    
    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "fake_probability": result["fake_probability"],
        "real_probability": result["real_probability"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 