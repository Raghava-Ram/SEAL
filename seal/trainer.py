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
        # Initialize variables to avoid reference errors in exception handling
        input_ids = attention_mask = labels_tensor = None
        
        # Check if texts or labels are empty (handle both lists and tensors)
        texts_empty = len(texts) == 0 if not isinstance(texts, torch.Tensor) else texts.numel() == 0
        labels_empty = len(labels) == 0 if not isinstance(labels, torch.Tensor) else labels.numel() == 0
        
        if texts_empty or labels_empty:
            raise ValueError("Texts and labels cannot be empty")
            
        # Ensure all labels are valid (within model's num_labels range)
        # Convert tensors to lists for iteration
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        if isinstance(texts, torch.Tensor):
            texts = texts.tolist()
        
        num_labels = getattr(self.model.config, "num_labels", 2)
        valid_labels = []
        valid_texts = []
        
        for text, label in zip(texts, labels):
            # Handle both regular values and tensor scalars
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            # Try to convert label to integer (handle strings, floats, etc.)
            try:
                label_int = int(float(label))
            except (ValueError, TypeError):
                # Skip invalid labels
                continue
            
            # Accept integer labels in [0, num_labels-1]
            if label_int >= 0 and label_int < num_labels:
                valid_labels.append(label_int)
                valid_texts.append(str(text))
            # For debugging: log when labels are out of range
            elif not hasattr(self, '_label_range_warned'):
                print(f"Warning: Label {label_int} out of range [0, {num_labels-1}]. Model has {num_labels} classes.")
                self._label_range_warned = True
                
        if not valid_texts:
            # More informative warning
            sample_labels = [l for l in labels[:5]]  # Show first 5 labels
            print(f"Warning: No valid labels found in batch. Sample labels: {sample_labels}, Model num_labels: {num_labels}")
            return 0.0
            
        # Ensure model is in training mode and on correct device with correct dtype
        self.model.train()
        self.model = self.model.to(self.device).float()  # Ensure model is in float32
        self.model.zero_grad()  # Clear any existing gradients
            
        # Ensure all parameters are float32 and require gradients
        for param in self.model.parameters():
            if param.is_floating_point():
                param.data = param.data.float()
            param.requires_grad_(True)
            
        try:
            
            # Tokenize the input texts
            inputs = self.tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # Move tensors to device and ensure correct dtypes
            input_ids = inputs['input_ids'].to(self.device, dtype=torch.long)
            attention_mask = inputs['attention_mask'].to(self.device, dtype=torch.long)
            
            # Convert labels to tensor (long for classification)
            labels_tensor = torch.tensor(valid_labels, device=self.device, dtype=torch.long)
            
            # Debug info for first batch
            if not hasattr(self, '_debug_printed'):
                print("\n=== Model and Batch Info ===")
                print(f"Model device: {next(self.model.parameters()).device}")
                print(f"Input types - input_ids: {input_ids.dtype}, attention_mask: {attention_mask.dtype}")
                print(f"Input shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
                
                # Print parameter info
                total_params = 0
                trainable_params = 0
                param_dtypes = {}
                
                for name, param in self.model.named_parameters():
                    total_params += param.numel()
                    if param.requires_grad:
                        trainable_params += param.numel()
                    
                    # Track parameter dtypes and gradient status
                    dtype_str = str(param.dtype)
                    grad_str = "grad" if param.requires_grad else "no_grad"
                    key = f"{dtype_str} ({grad_str})"
                    param_dtypes[key] = param_dtypes.get(key, 0) + 1
                
                print(f"\n=== Parameter Summary ===")
                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params/max(1, total_params):.1f}%)")
                print("\nParameter dtypes and gradient status:")
                for dtype, count in param_dtypes.items():
                    print(f"  {dtype}: {count:,} parameters")
                
                print("\n=== Model Architecture ===")
                print(self.model)
                print("\n" + "="*50 + "\n")
                self._debug_printed = True
            
            try:
                # Forward pass - don't pass labels to let the model handle the forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # Get logits and compute loss
                logits = outputs.logits  # Shape: [batch_size, num_classes]
                num_classes = logits.shape[1] if logits.dim() > 1 else 1

                if num_classes > 1:
                    # Multi-class classification: use cross entropy
                    loss = torch.nn.functional.cross_entropy(
                        logits,
                        labels_tensor
                    )
                else:
                    # Binary/regression fallback: MSE on squeezed logits
                    loss = torch.nn.functional.mse_loss(
                        logits.view(-1),
                        labels_tensor.float().view(-1),
                        reduction='mean'
                    )
                
                # Ensure loss is float32
                loss = loss.float()
                
                # Print loss info for debugging
                if not hasattr(self, '_loss_printed'):
                    print(f"\n=== Loss Info ===")
                    print(f"Loss: {loss.item():.4f}")
                    print(f"Loss type: {loss.dtype}, shape: {loss.shape}")
                    print(f"Gradient function: {loss.grad_fn}")
                    self._loss_printed = True
                    
            except Exception as e:
                print("\n=== Forward Pass Error ===")
                print(f"Input shapes - input_ids: {input_ids.shape if hasattr(input_ids, 'shape') else type(input_ids)}")
                print(f"attention_mask: {attention_mask.shape if hasattr(attention_mask, 'shape') else type(attention_mask)}")
                print(f"labels: {labels_tensor.shape if hasattr(labels_tensor, 'shape') else type(labels_tensor)}")
                print(f"Model device: {next(self.model.parameters()).device}")
                print(f"Input device: {input_ids.device if hasattr(input_ids, 'device') else 'N/A'}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                
                # Print model parameter info
                print("\n=== Model Parameter Info ===")
                for name, param in self.model.named_parameters():
                    print(f"{name}: {param.dtype} on {param.device}, requires_grad={param.requires_grad}, shape={tuple(param.shape)}")
                
                raise RuntimeError(f"Forward pass failed: {str(e)}") from e
            
            try:
                # Backward pass
                loss.backward()
                
                # Debug: Check gradients
                if not hasattr(self, '_grad_printed'):
                    print("\n=== Gradient Info ===")
                    grad_info = {}
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            grad_info[name] = {
                                'dtype': str(param.grad.dtype),
                                'device': str(param.grad.device),
                                'norm': f"{grad_norm:.6f}",
                                'shape': tuple(param.grad.shape)
                            }
                    
                    # Print gradient summary
                    print(f"Parameters with gradients: {len(grad_info)}/{len(list(self.model.parameters()))}")
                    for name, info in list(grad_info.items())[:5]:  # Print first 5 for brevity
                        print(f"  {name}: {info}")
                    if len(grad_info) > 5:
                        print(f"  ... and {len(grad_info) - 5} more parameters with gradients")
                    self._grad_printed = True
                
                # Clip gradients if max_grad_norm is set
                max_grad_norm = getattr(self, 'max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=max_grad_norm,
                        norm_type=2.0
                    )
                
                # Step the optimizer
                self.optimizer.step()
                
                # Update learning rate scheduler if provided
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()
                
                return loss.item()
                
            except Exception as e:
                print("\n=== Backward Pass Error ===")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                
                # Print gradient information
                print("\n=== Gradient Status ===")
                grad_count = 0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad_count += 1
                        print(f"{name}: {param.grad.dtype} on {param.grad.device}, norm={param.grad.norm().item():.6f}")
                print(f"\nParameters with gradients: {grad_count}/{len(list(self.model.parameters()))}")
                
                # Print model info
                print("\n=== Model Info ===")
                print(f"Model device: {next(self.model.parameters()).device}")
                print(f"Model training mode: {self.model.training}")
                
                raise RuntimeError(f"Backward pass failed: {str(e)}") from e
            
        except Exception as e:
            # Print detailed error information
            print("\n" + "="*50)
            print("=== Error Details ===")
            print(f"Input IDs shape: {input_ids.shape if hasattr(input_ids, 'shape') else type(input_ids)}, dtype: {input_ids.dtype if hasattr(input_ids, 'dtype') else 'N/A'}")
            print(f"Attention mask shape: {attention_mask.shape if hasattr(attention_mask, 'shape') else type(attention_mask)}, dtype: {attention_mask.dtype if hasattr(attention_mask, 'dtype') else 'N/A'}")
            # Use labels_tensor if available, otherwise check labels parameter
            if labels_tensor is not None and hasattr(labels_tensor, 'shape'):
                print(f"Labels tensor shape: {labels_tensor.shape}, dtype: {labels_tensor.dtype}")
            elif isinstance(labels, torch.Tensor):
                print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
            elif isinstance(labels, list):
                print(f"Labels type: list, length: {len(labels)}")
            else:
                print(f"Labels type: {type(labels)}")
            print(f"Model device: {next(self.model.parameters()).device}")
            print(f"Model dtype: {next(self.model.parameters()).dtype}")
            
            # Check parameter dtypes
            param_dtypes = {}
            for name, param in self.model.named_parameters():
                param_dtypes[param.dtype] = param_dtypes.get(param.dtype, 0) + 1
                if param.dtype != torch.float32:
                    print(f"Parameter {name} has dtype {param.dtype} on {param.device}")
            
            print(f"\nParameter dtype counts: {param_dtypes}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\n" + "="*50 + "\n")
            
            # Print model's forward signature for debugging
            if hasattr(self.model, 'forward'):
                import inspect
                sig = inspect.signature(self.model.forward)
                print("\nModel forward signature:")
                print(sig)
            
            # Re-raise the exception to stop execution
            raise RuntimeError(f"Training failed: {str(e)}") from e

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
                max_length=256,  # Reduced from 512 to 256 for faster processing
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
