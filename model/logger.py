import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingLogger:
    def __init__(self, log_dir=None):
        """
        Initialize the training logger
        
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        if log_dir is None:
            log_dir = Path("model/logs")
        
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logs will be saved to {log_dir}")
    
    def log_metrics(self, metrics, step):
        """
        Log metrics to TensorBoard
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step/epoch number
        """
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def log_model_graph(self, model, input_tensor):
        """
        Log model graph to TensorBoard
        
        Args:
            model: PyTorch model
            input_tensor: Example input tensor
        """
        try:
            self.writer.add_graph(model, input_tensor)
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")
    
    def log_confusion_matrix(self, cm, step):
        """
        Log confusion matrix as an image to TensorBoard
        
        Args:
            cm: Confusion matrix
            step: Current step/epoch number
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Epoch {step})')
        
        self.writer.add_figure('Confusion Matrix', plt.gcf(), step)
        plt.close()
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()

def get_logger(experiment_name="fake_news_detection"):
    """
    Get a training logger instance
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        TrainingLogger: A training logger instance
    """
    log_dir = Path(f"model/logs/{experiment_name}")
    return TrainingLogger(log_dir=log_dir) 