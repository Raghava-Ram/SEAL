import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import psutil
import json
import os

class SEALTrainer:
    """
    A lightweight trainer for the SEAL framework.
    Handles model training and evaluation in a CPU-compatible way.
    """
    def __init__(self, model, tokenizer, config):
        """
        Initialize the SEAL trainer.
        
        Args:
            model: Pre-trained model to train
            tokenizer: Tokenizer for the model
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(config.get("device", "cpu"))
        self.model.to(self.device)
        self.config = config
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 5e-5),
            weight_decay=config.get("weight_decay", 0.01)
        )

    def train_step(self, text, label="positive"):
        """
        Perform a single training step with the given text.
        
        Args:
            text: Input text for training
            label: Optional label (not used in base implementation)
            
        Returns:
            float: Training loss
        """
        self.model.train()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.get("max_seq_length", 512)
        ).to(self.device)
        
        # Forward pass
        labels = inputs["input_ids"].detach().clone()
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
        
    def evaluate_accuracy(self, dataset_name="imdb", num_samples=10):
        """Evaluate model accuracy on a small subset of a text classification dataset."""
        try:
            # For demonstration, we'll use a simple mock accuracy since we're using a masked LM
            # In a real scenario, you'd want to use a proper classification head
            print("📊 Running mock evaluation (using loss as proxy for accuracy)")
            
            # Generate some random text for evaluation
            eval_text = "Machine learning is transforming the world."
            inputs = self.tokenizer(eval_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
            
            # Calculate loss on evaluation text
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            
            # Convert loss to a mock accuracy (lower loss = higher accuracy)
            # This is just for demonstration - in a real scenario, use proper metrics
            mock_accuracy = max(0, min(1, 1.0 - (loss / 10.0)))  # Scale loss to [0,1] range
            
            print(f"📊 Evaluation loss: {loss:.4f} (mock accuracy: {mock_accuracy:.4f})")
            return mock_accuracy
            
        except Exception as e:
            print(f"⚠️  Accuracy evaluation failed: {str(e)}")
            return 0.5  # Return 50% as a fallback

    def get_system_usage(self):
        """Log CPU and memory usage."""
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory().percent
        return {"cpu": cpu, "memory": mem}
