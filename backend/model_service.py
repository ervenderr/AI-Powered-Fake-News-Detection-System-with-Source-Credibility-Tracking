import sys
import os
from pathlib import Path

# Add the project root to the path so we can import from the model directory
sys.path.append(str(Path(__file__).parent.parent))

from model.inference import FakeNewsDetector
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

class ModelService:
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one model is loaded"""
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance.detector = None
        return cls._instance
    
    def load_model(self):
        """Load the fake news detection model"""
        try:
            logger.info("Loading fake news detection model...")
            self.detector = FakeNewsDetector()
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_prediction(self, text):
        """
        Get prediction for the given text
        
        Args:
            text: The news text to classify
            
        Returns:
            dict: A dictionary containing the prediction results
        """
        if self.detector is None:
            success = self.load_model()
            if not success:
                return {
                    "error": "Failed to load model",
                    "status": "error"
                }
        
        try:
            result = self.detector.predict(text)
            return {
                "prediction": result["label"],
                "confidence": result["probability"],
                "fake_probability": result["fake_probability"],
                "real_probability": result["real_probability"],
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                "error": str(e),
                "status": "error"
            }

# Initialize the model service
model_service = ModelService()

# Attempt to load the model at module import time
try:
    model_service.load_model()
except Exception as e:
    logger.warning(f"Could not load model at startup: {e}")
    logger.warning("Model will be loaded on first prediction request") 