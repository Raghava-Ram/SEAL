import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import psutil
import json
import os
import random
from typing import List, Dict, Any, Tuple
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
        
    def predict_with_confidence(self, text):
        """
        Get model prediction and confidence for a single text input.
        
        Args:
            text: Input text to predict on
            
        Returns:
            tuple: (predicted_label, confidence_score)
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            out = self.model(**inputs)
            probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()

        # Get prediction and confidence
        pred_idx = int(probs.argmax())
        conf = float(probs[pred_idx])
        label = "positive" if pred_idx == 1 else "negative"
        return label, conf

    def train_on_batch(self, texts: List[str], labels: List[int]) -> float:
        """
        Train on a batch of examples.
        
        Args:
            texts: List of input texts
            labels: List of integer labels (0 or 1)
            
        Returns:
            float: Training loss
            
        Raises:
            ValueError: If no valid texts or labels are provided
        """
        if not texts or not labels:
            raise ValueError("Texts and labels cannot be empty")
            
        # Ensure all labels are valid (0 or 1)
        valid_labels = []
        valid_texts = []
        for text, label in zip(texts, labels):
            if label in [0, 1]:
                valid_labels.append(label)
                valid_texts.append(text)
                
        if not valid_texts:
            raise ValueError("No valid labels found in batch")
            
        self.model.train()
        device = next(self.model.parameters()).device
        
        try:
            # Tokenize inputs
            inputs = self.tokenizer(
                valid_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512  # Add max_length to prevent very long sequences
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Convert labels to tensor
            labels_tensor = torch.tensor(valid_labels, dtype=torch.long, device=device)

            # Forward and backward pass
            out = self.model(**inputs, labels=labels_tensor)
            loss = out.loss

            if loss is not None:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                return float(loss.item())
            return 0.0
            
        except Exception as e:
            print(f"Error in train_on_batch: {str(e)}")
            return 0.0  # Return 0 loss on error to continue training

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer': self.tokenizer,
            'config': self.config
        }, path)
        print(f"✅ Checkpoint saved to {path}")
        
    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        """
        Make predictions on a list of input texts.
        
        Args:
            texts: List of input text strings
            batch_size: Batch size for prediction
            
        Returns:
            List of predicted class indices
        """
        self.model.eval()
        predictions = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize and prepare batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(batch_preds.tolist())
        
        return predictions

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
        """
        Compute accuracy on provided dataset.
        
        Args:
            dataset: List of dicts with 'text' and 'label' keys
            
        Returns:
            float: Accuracy score between 0 and 1
        """
        if not dataset:
            print("⚠️  Empty dataset provided for evaluation")
            return 0.0
            
        self.model.eval()
        correct = 0
        total = 0
        
        for i, ex in enumerate(dataset):
            if i >= 100:  # Evaluate on max 100 samples for speed
                break
                
            text, true_label = ex.get("text", ""), ex.get("label", 0)
            if not text:
                continue
                
            try:
                # For local mode, text is already the prediction ("positive" or "negative")
                if isinstance(text, str) and text.lower() in ["positive", "negative"]:
                    pred = 1 if text.lower() == "positive" else 0
                    correct += 1 if pred == true_label else 0
                    total += 1
                    continue
                    
                # For LLM mode or other cases where we need to make a prediction
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
                    correct += 1 if pred == true_label else 0
                    total += 1
                    
            except Exception as e:
                print(f"⚠️  Error processing sample {i}: {str(e)}")
                continue
        
        if total == 0:
            print("⚠️  No valid predictions made")
            return 0.0
            
        accuracy = correct / total
        print(f"📊 Accuracy on {total} samples: {accuracy:.4f} ({correct}/{total})")
        return accuracy

    def get_system_usage(self):
        """Log CPU and memory usage."""
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory().percent
        return {"cpu": cpu, "memory": mem}
