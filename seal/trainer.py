import torch
from torch.optim import AdamW

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
            Model outputs
        """
        self.model.eval()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.get("max_seq_length", 512)
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs
