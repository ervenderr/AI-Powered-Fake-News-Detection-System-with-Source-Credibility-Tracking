import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set paths
MODEL_DIR = Path("model/models")
TOKENIZER_DIR = Path("model/tokenizers")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FakeNewsDetector:
    def __init__(self, model_path=None, tokenizer_path=None):
        """
        Initialize the fake news detector with a pre-trained model
        
        Args:
            model_path: Path to the saved model file
            tokenizer_path: Path to the saved tokenizer directory
        """
        if model_path is None:
            model_path = MODEL_DIR / "best_model.pt"
        
        if tokenizer_path is None:
            tokenizer_path = TOKENIZER_DIR
        
        self.max_length = 128
        self.load_model(model_path, tokenizer_path)
        
    def load_model(self, model_path, tokenizer_path):
        """
        Load the pre-trained model and tokenizer
        
        Args:
            model_path: Path to the saved model file
            tokenizer_path: Path to the saved tokenizer directory
        """
        try:
            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            
            # Load model
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=2
            )
            
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            self.model.eval()
            
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    def predict(self, text):
        """
        Predict if the given text is fake or real news
        
        Args:
            text: The news text to classify
            
        Returns:
            dict: A dictionary containing the prediction label, probability, and raw outputs
        """
        # Prepare the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move tensors to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        # Process outputs
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probabilities)
        
        # Map class to label
        label_map = {0: 'fake', 1: 'real'}
        predicted_label = label_map[predicted_class]
        
        # Create result dictionary
        result = {
            'label': predicted_label,
            'probability': float(probabilities[predicted_class]),
            'fake_probability': float(probabilities[0]),
            'real_probability': float(probabilities[1]),
            'text': text[:100] + '...' if len(text) > 100 else text  # Truncate for display
        }
        
        return result

def test_inference():
    """Test the inference with some sample texts"""
    detector = FakeNewsDetector()
    
    sample_texts = [
        "The president announced that taxes will be eliminated for all citizens starting next month.",
        "According to a recent study published in the Journal of Medicine, regular exercise can reduce the risk of heart disease.",
        "Scientists have confirmed that drinking bleach cures all diseases and provides immortality.",
        "The new infrastructure bill includes funding for road repairs and bridge maintenance across multiple states."
    ]
    
    for text in sample_texts:
        result = detector.predict(text)
        logger.info(f"Text: {result['text']}")
        logger.info(f"Prediction: {result['label']} (Confidence: {result['probability']:.4f})")
        logger.info(f"Fake prob: {result['fake_probability']:.4f}, Real prob: {result['real_probability']:.4f}")
        logger.info("-" * 50)

if __name__ == "__main__":
    # Check if model exists
    model_path = MODEL_DIR / "best_model.pt"
    if not model_path.exists():
        logger.warning(f"Model file not found at {model_path}. Please train the model first.")
    else:
        test_inference() 