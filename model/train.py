import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import time
from model.logger import get_logger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set paths
MODEL_DIR = Path("model/models")
TOKENIZER_DIR = Path("model/tokenizers")
DATA_DIR = Path("data/processed")

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True, parents=True)
TOKENIZER_DIR.mkdir(exist_ok=True, parents=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.statement
        self.targets = dataframe.label_num
        self.max_length = max_length

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        target = self.targets.iloc[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def load_data():
    """Load the LIAR dataset"""
    train_df = pd.read_csv(DATA_DIR / "liar_train.csv")
    valid_df = pd.read_csv(DATA_DIR / "liar_valid.csv")
    test_df = pd.read_csv(DATA_DIR / "liar_test.csv")
    
    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Validation data shape: {valid_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    
    return train_df, valid_df, test_df

def create_data_loaders(train_df, valid_df, test_df, tokenizer, batch_size=16):
    """Create PyTorch DataLoaders for train, validation and test sets"""
    train_dataset = FakeNewsDataset(
        dataframe=train_df,
        tokenizer=tokenizer
    )
    
    valid_dataset = FakeNewsDataset(
        dataframe=valid_df,
        tokenizer=tokenizer
    )
    
    test_dataset = FakeNewsDataset(
        dataframe=test_df,
        tokenizer=tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, valid_loader, test_loader

def train_epoch(model, data_loader, optimizer, scheduler, device):
    """Train the model for one epoch"""
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        
        correct_predictions += torch.sum(preds == targets)
        total_predictions += targets.shape[0]
        
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    accuracy = correct_predictions.double() / total_predictions
    return np.mean(losses), accuracy.item()

def eval_model(model, data_loader, device):
    """Evaluate the model on validation or test data"""
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            
            correct_predictions += torch.sum(preds == targets)
            total_predictions += targets.shape[0]
            
            losses.append(loss.item())
            
            all_predictions.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
    
    accuracy = correct_predictions.double() / total_predictions
    f1 = f1_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)
    
    return np.mean(losses), accuracy.item(), f1, cm

def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix and save to file"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def save_metrics(train_losses, val_losses, train_accs, val_accs, output_path):
    """Save training metrics to file"""
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    """Plot training curves and save to files"""
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir / 'loss_curve.png')
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(output_dir / 'accuracy_curve.png')
    plt.close()

def main():
    # Hyperparameters
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    MODEL_NAME = 'bert-base-uncased'
    
    # Load data
    train_df, valid_df, test_df = load_data()
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # Binary classification
        output_attentions=False,
        output_hidden_states=False
    )
    
    model.to(device)
    
    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        train_df, valid_df, test_df, tokenizer, BATCH_SIZE
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Initialize TensorBoard logger
    tb_logger = get_logger(experiment_name=f"fake_news_bert_{time.strftime('%Y%m%d_%H%M%S')}")
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_f1 = 0
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, val_f1, val_cm = eval_model(model, valid_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Log metrics to TensorBoard
        tb_logger.log_metrics({
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'val/f1': val_f1
        }, epoch)
        
        # Log confusion matrix
        tb_logger.log_confusion_matrix(val_cm, epoch)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            logger.info(f"Saving best model with F1: {val_f1:.4f}")
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            tokenizer.save_pretrained(TOKENIZER_DIR)
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_acc, test_f1, test_cm = eval_model(model, test_loader, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    
    # Log test metrics
    tb_logger.log_metrics({
        'test/loss': test_loss,
        'test/accuracy': test_acc,
        'test/f1': test_f1
    }, EPOCHS)
    
    # Log test confusion matrix
    tb_logger.log_confusion_matrix(test_cm, EPOCHS)
    
    # Save confusion matrix
    plot_confusion_matrix(test_cm, MODEL_DIR / "confusion_matrix.png")
    
    # Save metrics
    save_metrics(train_losses, val_losses, train_accs, val_accs, MODEL_DIR / "metrics.json")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, MODEL_DIR)
    
    # Close TensorBoard logger
    tb_logger.close()
    
    logger.info("Training and evaluation completed!")

if __name__ == "__main__":
    main() 