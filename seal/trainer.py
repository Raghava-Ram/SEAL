import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import psutil
import json
import os
import random
from transformers import AutoModelForSequenceClassification

class SEALTrainer:
    """
    A lightweight trainer for the SEAL framework.
    Handles model training and evaluation in a CPU-compatible way.
    """
    def __init__(self, model, tokenizer, config):
        """
        Initialize the SEAL trainer.
        
        Args:
            model: The model to train
            tokenizer: Tokenizer for the model
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cpu")  # Force CPU for consistency
        self.model.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 5e-5),
            weight_decay=config.get("weight_decay", 0.01)
        )

    def train_step(self, text, label=1):
        """
        Perform a single training step with the given text and label.
        
        Args:
            text: Input text for training
            label: Numeric label (0 or 1) for the input text (default: 1 for positive)
            
        Returns:
            Loss value for the training step
        """
        self.model.train()
        
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Convert label to tensor
        labels = torch.tensor([label], dtype=torch.long).to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()

    def evaluate(self, text):
        """
        Evaluate the model on the given text.
        
        Args:
            text: Input text for evaluation
            
        Returns:
            Model outputs for the given input text
        """
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs
        
    def load_imdb(self, subset_size=500):
        """Load a small subset of IMDB dataset for CPU testing."""
        try:
            ds = load_dataset("imdb", split="train[:5%]").shuffle(seed=42)
            # Downsample for speed
            small_ds = [{"text": ex["text"], "label": ex["label"]}
                       for ex in ds.select(range(min(subset_size, len(ds))))]
            print(f"📚 Loaded {len(small_ds)} IMDB samples.")
            return small_ds
        except Exception as e:
            print(f"⚠️  Failed to load IMDB dataset: {str(e)}")
            return []
    
    def evaluate_accuracy(self, dataset):
        """Compute accuracy on provided dataset."""
        if not dataset:
            print("⚠️  Empty dataset provided for evaluation")
            return 0.0
            
        self.model.eval()
        preds, labels = [], []
        
        for i, ex in enumerate(dataset):
            if i >= 100:  # Evaluate on max 100 samples for speed
                break
                
            text, label = ex.get("text", ""), ex.get("label", 0)
            if not text:
                continue
                
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    pred = torch.argmax(outputs.logits, dim=-1).item()
                    preds.append(pred)
                    labels.append(label)
                    
            except Exception as e:
                print(f"⚠️  Error processing sample {i}: {str(e)}")
                continue
        
        if not preds:
            print("⚠️  No valid predictions made")
            return 0.0
            
        acc = accuracy_score(labels, preds)
        print(f"📊 Accuracy on {len(preds)} samples: {acc:.4f}")
        return acc

    def get_system_usage(self):
        """Log CPU and memory usage."""
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory().percent
        return {"cpu": cpu, "memory": mem}
