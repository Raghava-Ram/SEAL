"""
SEAL Runner Module

Coordinates the SEAL self-adaptive learning loop, including model training and evaluation.
"""
import time
import os
import json
import random
import torch
import numpy as np
from statistics import mean
from collections import Counter
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import yaml
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from seal.llm_adapter import get_llm_client
from seal.prompt_adapter import create_prompt
from seal.trainer import SEALTrainer
from seal.adapter import generate_edit
from seal.memory import EditCache
from seal.utility import score_edit_simple
from seal.replay import mix_batches

def run_imdb_comparison(config_path: str = "configs/default.yaml") -> None:
    """
    Compare SEAL performance using local vs LLM edits on the IMDB dataset.
    
    Args:
        config_path: Path to the configuration file
    """
    print("🚀 Starting SEAL IMDB Comparison\n")
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    hybrid_mode = (isinstance(config, dict) and ((config.get('phase') == 'hybrid') or config.get('hybrid', False)))
    
    # Create output directory
    output_dir = config.get("save_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize LLM client
    print("🔄 Initializing LLM client...")
    try:
        llm = get_llm_client(config)
        print(f"✅ Initialized {config['llm']['backend']} LLM client.\n")
    except Exception as e:
        print(f"❌ Failed to initialize LLM client: {str(e)}")
        return
    
    # Load dataset
    print("📥 Loading IMDB dataset...")
    try:
        dataset = load_dataset("imdb", split="train")
        # Convert to list of dicts for easier processing
        imdb_data = [{"text": ex["text"], "label": ex["label"]} for ex in dataset]
        print(f"✅ Loaded {len(imdb_data)} IMDB samples.\n")
    except Exception as e:
        print(f"❌ Failed to load IMDB dataset: {str(e)}")
        return
    
    comparison_results = {}
    edit_modes = config.get("edit_modes", ["local", "llm"])
    
    # Process each edit mode
    for mode in edit_modes:
        if mode == "llm" and not hasattr(llm, 'generate'):
            print(f"⚠️ Skipping {mode} mode - LLM client not properly initialized")
            continue
            
        print(f"\n🔍 Processing {mode.upper()} mode...")
        print(f"\n{'='*50}")
        print(f"🧠 Running SEAL in {mode.upper()} mode")
        print(f"{'='*50}")
        
        # Initialize model and tokenizer
        model_name = config.get("model_name", "distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("🚀 Starting SEAL Multi-Task Learning")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load model with proper configuration
        print("\n=== Initializing Model ===")
        model_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
            torch_dtype=torch.float32  # Ensure config specifies float32
        )
        
        # Initialize model with explicit float32 precision
        print(f"Loading model with config: {model_config}")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=model_config,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying with from_config...")
            model = AutoModelForSequenceClassification.from_config(model_config)
        
        # Ensure model is in float32 and on the correct device
        model = model.to(device).float()
        print(f"Model loaded on {device} with dtype {next(model.parameters()).dtype}")
        
        # Reinitialize the classifier with proper initialization
        if hasattr(model, 'classifier'):
            in_features = model.classifier.in_features if hasattr(model.classifier, 'in_features') else model.config.hidden_size
            print(f"Reinitializing classifier with in_features={in_features}, out_features=2")
            
            # Create new classifier with explicit float32 weights
            new_classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(in_features, 2, dtype=torch.float32)
            ).to(device)
            
            # Initialize weights with proper scaling
            for module in new_classifier.modules():
                if isinstance(module, nn.Linear):
                    print(f"Initializing {module} weights")
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            
            model.classifier = new_classifier
        
        # Double-check all parameters are float32 and on the right device
        print("\n=== Verifying Parameter Types ===")
        param_dtypes = {}
        for name, param in model.named_parameters():
            if param.dtype != torch.float32:
                print(f"Converting {name} from {param.dtype} to float32")
                param.data = param.data.to(torch.float32)
            if param.device != device:
                print(f"Moving {name} to {device}")
                param.data = param.data.to(device)
            # Track parameter dtypes
            param_dtypes[param.dtype] = param_dtypes.get(param.dtype, 0) + 1
        
        print(f"\nParameter dtype distribution: {param_dtypes}")
        
        # Print model summary (skip verbose architecture printing in hybrid mode)
        print("\n=== Model Summary ===")
        print(f"Device: {next(model.parameters()).device}")
        print(f"Dtype: {next(model.parameters()).dtype}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        if not hybrid_mode:
            # Print model architecture only when not running Phase 3 hybrid (reduces verbosity)
            print("\n=== Model Architecture ===")
            print(model)
            print("\n" + "="*50 + "\n")
        
        # Verify model forward pass with dummy input
        try:
            print("Verifying model forward pass...")
            with torch.no_grad():
                dummy_input = {
                    'input_ids': torch.randint(0, 100, (2, 16), device=device, dtype=torch.long),
                    'attention_mask': torch.ones((2, 16), device=device, dtype=torch.long)
                }
                output = model(**dummy_input)
                print(f"Forward pass successful! Output shape: {output.logits.shape if hasattr(output, 'logits') else 'N/A'}")
        except Exception as e:
            print(f"Error during forward pass check: {e}")
        
        print("\n" + "="*50 + "\n")
        
        return model
        
        # Ensure all parameters are float32 and on the right device
        for name, param in model.named_parameters():
            if param.dtype != torch.float32:
                print(f"Converting {name} from {param.dtype} to float32")
                param.data = param.data.float()
            if param.device != device:
                param.data = param.data.to(device)
        
        # Set model to train mode
        model.train()
        
        # Print model info
        print("\n=== Model Configuration ===")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print("=========================\n")
        
        # Print model summary
        print("\n=== Model Summary ===")
        print(f"Device: {next(model.parameters()).device}")
        print(f"Dtype: {next(model.parameters()).dtype}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print("===================\n")
        
        # Print model device and dtype for debugging
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        # Set model to train mode
        model.train()
        
        # Verify all parameters are float32
        for name, param in model.named_parameters():
            if param.dtype != torch.float32:
                print(f"Converting {name} from {param.dtype} to float32")
                param.data = param.data.float()
        
        # Initialize trainer and ensure model is float32
        model = model.to(torch.float32)
        trainer = SEALTrainer(model, tokenizer, config)
        
        # Initialize metrics
        loss_history = []
        accuracy_history = []
        cpu_log = []
        mem_log = []
        
        # Run training loop
        max_steps = config.get("max_steps", 10)
        log_interval = config.get("log_interval", 2)
        
        for step in range(max_steps):
            # Sample a random example
            sample = random.choice(imdb_data)
            
            # Generate edit using the appropriate method
            if mode == "local":
                # Get the opposite sentiment for the edit
                target_label = 1 - sample["label"]  # Flip the label for the edit
                edited_text = generate_edit(sample["text"], target_label=target_label, mode="local")
                
                # Train step with the edited text and target label
                loss = trainer.train_step(edited_text, label=target_label)
                loss_history.append(loss)
                
                # Log progress
                print(f"📊 Step {step+1}/{max_steps} - Loss: {loss:.4f}")
                
                # Evaluate periodically
                if (step + 1) % log_interval == 0 or step == max_steps - 1:
                    # Evaluate on a small subset for efficiency
                    eval_subset = random.sample(imdb_data, min(100, len(imdb_data)))
                    accuracy = trainer.evaluate_accuracy(eval_subset)
                    accuracy_history.append(accuracy)
                    print(f"   ✅ Accuracy: {accuracy:.4f}")
                
                # Log system usage
                usage = trainer.get_system_usage()
                cpu_log.append(usage["cpu"])
                mem_log.append(usage["memory"])
                print(f"   💻 CPU: {usage['cpu']:.1f}% | Memory: {usage['memory']:.1f}%")
                
            else:  # llm mode
                try:
                    # Create a prompt for sentiment analysis
                    prompt = create_prompt(
                        system_message="You are a helpful assistant that analyzes movie reviews. "
                                     "Classify the sentiment of the following review as positive or negative. "
                                     "Only respond with 'positive' or 'negative'.",
                        user_message=f"Review: {sample['text']}"
                    )
                    
                    # Get LLM prediction
                    response = llm.generate(prompt)
                    
                    # Simple post-processing to extract sentiment
                    response = response.lower().strip()
                    if "positive" in response:
                        edited_text = "positive"
                    elif "negative" in response:
                        edited_text = "negative"
                    else:
                        # Default to original label if we can't determine sentiment
                        edited_text = "positive" if sample["label"] == 1 else "negative"
                        
                except Exception as e:
                    print(f"⚠️ Error generating LLM edit: {str(e)}")
                    edited_text = "positive" if sample["label"] == 1 else "negative"
                
                # Train step
                loss = trainer.train_step(edited_text, label=sample["label"])
                loss_history.append(loss)
                
                # Log progress
                print(f"📊 Step {step+1}/{max_steps} - Loss: {loss:.4f}")
                
                # Evaluate periodically
                if (step + 1) % log_interval == 0 or step == max_steps - 1:
                    # Evaluate on a small subset for efficiency
                    eval_subset = random.sample(imdb_data, min(100, len(imdb_data)))
                    accuracy = trainer.evaluate_accuracy(eval_subset)
                    accuracy_history.append(accuracy)
                    print(f"   ✅ Accuracy: {accuracy:.4f}")
                
                # Log system usage
                usage = trainer.get_system_usage()
                cpu_log.append(usage["cpu"])
                mem_log.append(usage["memory"])
                print(f"   💻 CPU: {usage['cpu']:.1f}% | Memory: {usage['memory']:.1f}%")
        
        # Save results for this mode
        mode_results = {
            "loss": loss_history,
            "accuracy": accuracy_history,
            "cpu_usage": cpu_log,
            "memory_usage": mem_log,
            "avg_cpu": sum(cpu_log) / len(cpu_log) if cpu_log else 0,
            "avg_memory": sum(mem_log) / len(mem_log) if mem_log else 0,
            "final_accuracy": accuracy_history[-1] if accuracy_history else 0,
            "final_loss": loss_history[-1] if loss_history else 0
        }
        
        # Save mode-specific results
        output_file = os.path.join(output_dir, f"imdb_{mode}_results.json")
        with open(output_file, "w") as f:
            json.dump(mode_results, f, indent=2)
        print(f"\n💾 Saved {mode} results to {output_file}")
        
        # Plot metrics
        plt.figure(figsize=(10, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, 'b-', label='Training Loss')
        plt.title(f'Training Loss ({mode.upper()} Mode)')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy
        if accuracy_history:
            plt.subplot(1, 2, 2)
            x = [i * log_interval for i in range(len(accuracy_history))]
            plt.plot(x, accuracy_history, 'g-o', label='Accuracy')
            plt.title(f'Accuracy ({mode.upper()} Mode)')
            plt.xlabel('Step')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"imdb_{mode}_metrics.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"📈 Saved {mode} metrics plot to {plot_file}")
        
        comparison_results[mode] = mode_results
    
    # Generate comparison summary
    if len(comparison_results) > 1:
        summary = {
            "comparison_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models_tested": list(comparison_results.keys()),
            "results": {}
        }
        
        # Add metrics for each mode
        for mode, results in comparison_results.items():
            summary["results"][mode] = {
                "final_accuracy": results["final_accuracy"],
                "final_loss": results["final_loss"],
                "avg_cpu_usage": results["avg_cpu"],
                "avg_memory_usage": results["avg_memory"],
                "num_steps": len(results["loss"])
            }
        
        # Save comparison summary
        summary_file = os.path.join(output_dir, "imdb_comparison_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*50)
        print("📊 Comparison Summary:")
        for mode, metrics in summary["results"].items():
            print(f"\n{mode.upper()} Mode:")
            print(f"  Accuracy: {metrics['final_accuracy']:.2f}")
            print(f"  Avg. CPU Usage: {metrics['avg_cpu_usage']:.1f}%")
            print(f"  Avg. Memory Usage: {metrics['avg_memory_usage']:.1f}%")
        
        print(f"\n📝 Full comparison summary saved to: {summary_file}")
    
    print("\n✅ IMDB comparison completed successfully!")

def run_seal_loop(config_path="configs/default.yaml"):
    """
    Run the SEAL self-adaptive learning loop.
    
    Args:
        config_path: Path to the configuration file
    """
    print("🚀 Starting SEAL architecture run\n")

    # Load configuration
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️ Config file not found at {config_path}, using default settings")
        config = {}

    # Set default values
    model_name = config.get("model", {}).get("name", "distilbert-base-uncased")
    device = config.get("device", "cpu")
    edit_mode = config.get("editing", {}).get("mode", "local")
    
    print(f"🔧 Configuration:")
    print(f"- Model: {model_name}")
    print(f"- Device: {device}")
    print(f"- Edit Mode: {edit_mode}\n")

    try:
        # Initialize model and tokenizer
        print("🔄 Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # For binary sentiment classification
        )
        
        # Initialize trainer and load IMDB data
        trainer = SEALTrainer(model, tokenizer, config)
        print("📥 Loading IMDB dataset...")
        imdb_data = trainer.load_imdb(subset_size=500)
        
        if not imdb_data:
            print("❌ Failed to load IMDB data. Exiting...")
            return
        
        # Initialize tracking
        loss_history = []
        accuracy_history = []
        cpu_log = []
        mem_log = []
        
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # Get initial system usage
        initial_usage = trainer.get_system_usage()
        print(f"\n🔧 Initial System Usage - CPU: {initial_usage['cpu']}% | Memory: {initial_usage['memory']}%")
        
        # Initialize memory
        memory = EditCache(
            path=config["memory"]["path"],
            max_size=config["memory"]["max_size"]
        )
        
        # Run SEAL loop
        max_steps = config.get("trainer", {}).get("max_steps", 10)
        
        # Initial evaluation before any training
        print("\n🔍 Running initial evaluation...")
        initial_acc = trainer.evaluate_accuracy(imdb_data[:100])
        accuracy_history.append(initial_acc)
        print(f"📊 Initial Accuracy: {initial_acc:.4f}")
        
        for step in range(max_steps):
            print(f"\n🔁 SEAL Step {step+1}/{max_steps}")

            # 1. Sample IMDB example
            sample = random.choice(imdb_data)
            original = sample["text"]
            label = sample["label"]

            # 2. Generate edit (local or llm)
            edited = generate_edit(original, label=label)

            # 3. Get predictions before & after
            pred_before, conf_before = trainer.predict_with_confidence(original)
            pred_after, conf_after = trainer.predict_with_confidence(edited)

            # 4. Build edit record
            edit_obj = {
                "original": original,
                "edit": edited,
                "label": label,
                "pred_before": pred_before,
                "pred_after": pred_after,
                "conf_before": conf_before,
                "conf_after": conf_after
            }

            # 5. Utility score
            utility = score_edit_simple(edit_obj)
            edit_obj["utility"] = utility

            # 6. Store in memory
            memory.add_edit(edit_obj)

            # 7. Create batch using replay
            batch = mix_batches(
                new_examples=[{"text": edited, "label": label}],
                memory=memory,
                batch_size=config["replay"]["batch_size"],
                replay_fraction=config["replay"]["replay_fraction"],
                policy=config["replay"]["policy"],
                hybrid=hybrid_mode
            )

            texts = [b["text"] for b in batch]
            labels = [b["label"] for b in batch]

            # 8. Train on mixed batch
            loss = trainer.train_on_batch(texts, labels)
            
            # Update tracking
            loss_history.append(loss)
            
            # Log progress
            print(f"📉 Loss: {loss:.4f} | Utility: {utility:.4f}")

            # 9. Periodic evaluation
            if (step + 1) % config["trainer"].get("eval_interval", 10) == 0 or step == max_steps - 1:
                acc = trainer.evaluate_accuracy(imdb_data[:100])
                accuracy_history.append(acc)
                print(f"📊 Eval Accuracy: {acc:.4f}")
                
                # Log system usage
                usage = trainer.get_system_usage()
                cpu_log.append(usage['cpu'])
                mem_log.append(usage['memory'])
                print(f"💻 System Usage - CPU: {usage['cpu']}% | Memory: {usage['memory']}%")
                
            # Small delay for better readability
            time.sleep(1)
        
        # Prepare results
        results = {
            "loss": loss_history,
            "accuracy": accuracy_history,
            "cpu": cpu_log,
            "memory": mem_log,
            "avg_cpu": mean(cpu_log) if cpu_log else 0,
            "avg_memory": mean(mem_log) if mem_log else 0,
            "final_loss": loss_history[-1] if loss_history else None,
            "final_accuracy": accuracy_history[-1] if accuracy_history else None,
            "initial_accuracy": initial_acc,
            "iterations": max_steps
        }
        
        # Save results to JSON
        output_file = "outputs/imdb_eval_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        
        # Create a more detailed plot
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, 'b-', label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Accuracy
        plt.subplot(1, 2, 2)
        if len(accuracy_history) > 1:
            x = [i*2 for i in range(len(accuracy_history))]
            plt.plot(x, accuracy_history, 'g-o', label='Accuracy')
            plt.axhline(y=initial_acc, color='r', linestyle='--', label='Initial Accuracy')
            plt.title(f'Model Accuracy (Final: {accuracy_history[-1]:.2f})')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plot_file = "outputs/imdb_metrics.png"
        plt.savefig(plot_file)
        
        print("\n✅ SEAL IMDB evaluation completed successfully!")
        print(f"📊 Results saved to outputs/ directory:")
        print(f"   - Training metrics: {plot_file}")
        print(f"   - Raw data: {output_file}")
        print(f"\n📈 Performance Summary:")
        print(f"   - Initial Accuracy: {initial_acc:.4f}")
        if accuracy_history:
            print(f"   - Final Accuracy: {accuracy_history[-1]:.4f}")
        print(f"   - Avg CPU Usage: {mean(cpu_log):.1f}%")
        print(f"   - Avg Memory Usage: {mean(mem_log):.1f}%")
    
    except Exception as e:
        print(f"\n❌ Error during SEAL execution: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        return

    print("\n✅ SEAL architectural base loop executed successfully on CPU.")

def run_sequential_tasks(config_path: str = "configs/default.yaml") -> None:
    """
    Run sequential multi-task learning with task balancing and evaluation.
    
    Args:
        config_path: Path to the configuration file
    """
    print("🚀 Starting SEAL Multi-Task Learning\n")
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # PHASE MODE DETECTION
    # Phase 1: baseline -> no replay
    # PHASE 2: seal -> SEAL-enabled replay and real utility computation
    baseline_mode = False
    seal_mode = False
    hybrid_mode = False
    if isinstance(config, dict):
        baseline_mode = (config.get('phase') == 'baseline') or config.get('baseline', False)
        seal_mode = (config.get('phase') == 'seal') or config.get('seal', False)
        hybrid_mode = (config.get('phase') == 'hybrid') or config.get('hybrid', False)

    if baseline_mode:
        print("\n⚠️  Running Phase 1: Baseline (Replay Disabled)\n")
    if seal_mode:
        print("\n🔁 Running Phase 2: SEAL-enabled (Replay Enabled)\n")
    if hybrid_mode:
        # Print banner once for Phase 3
        print("\n🔁 Running Phase 3: Hybrid SEAL (LLM-guided + Light Replay)\n")
        # Exact required runtime message (print once)
        print("Running Phase 3: Hybrid SEAL (LLM-guided + Light Replay)")
    # Hybrid configuration: sparse LLM interval (default 10)
    hybrid_cfg = config.get('hybrid', {}) if isinstance(config, dict) else {}
    llm_interval = int(hybrid_cfg.get('llm_interval', 10))
    
    # Get task configuration
    task_config = config.get("tasks", {})
    task_order = task_config.get("order", ["imdb", "squad", "arc"])
    steps_per_task = task_config.get("steps", 300)
    eval_size = task_config.get("eval_size", 200)
    
    # Initialize components
    model_name = config.get("model_name", "distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2  # Will be updated for each task
    )
    
    # Initialize trainer and memory
    trainer = SEALTrainer(model, tokenizer, config)
    memory = EditCache()
    # Task-specific classifier heads storage
    import copy as _copy
    task_classifiers = {}
    
    # Track task-specific data and metrics
    task_datasets = {}
    task_val_sets = {}
    # Initialize task_metrics with empty lists for each task
    task_metrics = {task: [] for task in task_order}
    # Track which tasks have been evaluated at each step
    task_evaluations = {task: [] for task in task_order}
    
    # Load task datasets
    print("📥 Loading task datasets...")
    from seal.tasks import get_task_loader
    
    for task_name in task_order:
        try:
            loader = get_task_loader(task_name)
            # Get task-specific config including limit
            task_config = config.get('tasks', {}).get(task_name, {})
            task_limit = task_config.get('limit')
            
            # Load data with limit if specified
            task_data = loader(task_size=task_limit) if task_limit else loader()
            
            # Shuffle the data before splitting
            import random
            random.shuffle(task_data)
            
            # Split into train/val (80/20 split)
            split_idx = int(0.8 * len(task_data))
            if split_idx > 0:
                train_split = task_data[:split_idx]
                val_split = task_data[split_idx:]
                task_datasets[task_name] = train_split
                task_val_sets[task_name] = val_split
            else:
                # If not enough data for a proper split, use all for training
                task_datasets[task_name] = task_data
                task_val_sets[task_name] = task_data[:1]  # At least one example for validation

            # Log label distributions for train/val to help debug class imbalance
            try:
                train_labels = [ex["label"] for ex in task_datasets[task_name] if "label" in ex]
                val_labels = [ex["label"] for ex in task_val_sets[task_name] if "label" in ex]
                train_counts = Counter(train_labels)
                val_counts = Counter(val_labels)
                print(f"   ▶ {task_name} train/val sizes: {len(train_labels)}/{len(val_labels)}")
                print(f"   ▶ {task_name} train label dist (top): {train_counts.most_common(5)}")
                print(f"   ▶ {task_name} val   label dist (top): {val_counts.most_common(5)}")
            except Exception:
                pass
                
            print(f"   ✅ Loaded {len(task_datasets[task_name])} examples for {task_name}")
            
        except Exception as e:
            print(f"❌ Failed to load task {task_name}: {str(e)}")
            return
    
    # Initialize results dictionary for evaluation
    results = {task: {} for task in task_order}
    
    # Main training loop over tasks
    for task_idx, task_name in enumerate(task_order):
        print(f"\n{'='*50}")
        print(f"🎯 Training on task {task_idx+1}/{len(task_order)}: {task_name.upper()}")
        print(f"{'='*50}")
        print(f"🔍 DEBUG: Starting task {task_name} (index {task_idx})")
        print(f"🔍 DEBUG: task_datasets keys: {list(task_datasets.keys())}")
        print(f"🔍 DEBUG: task_val_sets keys: {list(task_val_sets.keys())}")
        
        # Get the dataset for the current task
        train_set = task_datasets[task_name]
        if not train_set:
            print(f"⚠️  No training data for task {task_name}, skipping...")
            continue
            
        print(f"📊 Training set size: {len(train_set)} examples")
        print(f"📊 Validation set size: {len(task_val_sets.get(task_name, []))} examples")
        print(f"📊 Task order: {task_order}")
        
        task_data = task_datasets[task_name]
        
        # Update model's classifier if needed (for tasks with different numbers of classes)
        # Calculate num_labels properly: for binary tasks, ensure 2; for multi-class, use max+1
        all_labels = [example["label"] for example in task_data]
        unique_labels = set(all_labels)
        max_label = max(all_labels) if all_labels else 0
        
        # Task-specific handling: IMDB and SQuAD are binary (2 classes)
        if task_name in ['imdb', 'squad']:
            num_labels = 2
        else:
            # For other tasks (like ARC), use max_label + 1 (since labels are 0-indexed)
            # But ensure we have at least as many classes as unique labels
            num_labels = max(max_label + 1, len(unique_labels))
        
        current_num_labels = model.num_labels if hasattr(model, 'num_labels') else model.config.num_labels
        
        if current_num_labels != num_labels:
            print(f"🔄 Updating classifier from {current_num_labels} to {num_labels} classes")
            
            # Create a new model with the correct number of labels
            model_config = AutoConfig.from_pretrained(
                model.config._name_or_path,
                num_labels=num_labels,
                torch_dtype=torch.float32
            )
            
            # Create a new model with the updated configuration
            device = model.device
            new_model = AutoModelForSequenceClassification.from_config(model_config).to(device)
            
            # Copy the base model weights
            new_model.distilbert = model.distilbert
            
            # Reinitialize classifier with the correct number of classes
            if hasattr(new_model, 'classifier') and hasattr(new_model, 'pre_classifier'):
                # Initialize classifier with the correct dimensions
                new_model.classifier = nn.Linear(
                    model_config.hidden_size,
                    num_labels
                ).to(device)
                
                # Initialize weights
                new_model.classifier.weight.data.normal_(mean=0.0, std=model_config.initializer_range)
                if new_model.classifier.bias is not None:
                    new_model.classifier.bias.data.zero_()
                    
                print(f"✅ Updated classifier to {num_labels} classes")
                
            # Update the model reference
            model = new_model
            trainer.model = model
            
            # Reinitialize optimizer with the new model parameters
            trainer.optimizer = AdamW(model.parameters(), lr=2e-5)
            
            print(f"   Classifier weights: {model.classifier.weight.shape if hasattr(model, 'classifier') else 'N/A'}")
            print(f"   Model device: {next(model.parameters()).device}")
            print(f"   Model dtype: {next(model.parameters()).dtype}")
            print(f"   Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
            # Don't skip the training loop - removed continue statement
        # Ensure model.config knows the correct num_labels for this task
        try:
            model.config.num_labels = num_labels
        except Exception:
            pass

        # Attach existing task-specific classifier if available, otherwise create a fresh one
        if task_name in task_classifiers:
            trainer.model.classifier = _copy.deepcopy(task_classifiers[task_name]).to(trainer.device)
        else:
            # Create a new classifier head for this task
            hidden_size = getattr(trainer.model.config, 'hidden_size', None) or getattr(trainer.model.config, 'dim', None)
            if hidden_size is None:
                hidden_size = trainer.model.config.hidden_size if hasattr(trainer.model.config, 'hidden_size') else 768
            new_clf = nn.Linear(hidden_size, num_labels).to(trainer.device)
            trainer.model.classifier = new_clf
            
        # Training loop for current task
        print(f"🔍 DEBUG: Starting training loop for {task_name} with {steps_per_task} steps")
        # PHASE 3: announce sparse LLM guidance once per task
        if hybrid_mode:
            print(f"PHASE 3: Using sparse LLM guidance (interval = {llm_interval})")
            # PHASE 3 EXTENSION: Task-aware replay weighting enabled (once per task)
            print("PHASE 3 EXTENSION: Task-aware replay weighting enabled")
        for step in range(steps_per_task):
            if step % 10 == 0 or step == steps_per_task - 1:
                print(f"🔍 DEBUG: Task {task_name} - Step {step+1}/{steps_per_task}")
            
            try:
                # Sample a batch
                batch_size = config.get('replay', {}).get('batch_size', 8)
                batch = random.sample(task_data, min(batch_size, len(task_data)))
                
                # Prepare batch data
                texts = [item["text"] for item in batch]
                labels = [item["label"] for item in batch]
                
                # Ensure labels are in the correct format
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=torch.long)
                
                # Train on the batch
                loss = trainer.train_on_batch(texts, labels)
                
                # Log progress
                if (step + 1) % 10 == 0 or step == steps_per_task - 1:
                    print(f"📊 Task {task_name} | Step {step+1}/{steps_per_task} | Loss: {loss:.4f}")
                    
            except Exception as e:
                print(f"⚠️  Error in training batch: {str(e)}")
                # Print more detailed error information
                import traceback
                print("\n=== Error Details ===")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("\n=== Stack Trace ===")
                traceback.print_exc()
                print("\n" + "="*50 + "\n")
                continue
            
            # Generate edits for the batch
            edited_batch = []
            for example in batch:
                try:
                    # For classification tasks, flip the label for editing
                    if task_name == "imdb":
                        target_label = 1 - example["label"]
                        previous_tasks = task_order[:task_idx]

                        # Sparse LLM usage only in hybrid mode
                        if hybrid_mode:
                            if llm_interval > 0 and (step % llm_interval) == 0:
                                # Use LLM-guided edit this step
                                edited_text = generate_edit(
                                    example["text"],
                                    target_label=target_label,
                                    mode='llm',
                                    current_task=task_name,
                                    previous_tasks=previous_tasks
                                )
                                print(f"PHASE 3: LLM-guided edit at step {step}")
                            else:
                                # Fallback to local edit for intermediate steps
                                edited_text = generate_edit(
                                    example["text"],
                                    target_label=target_label,
                                    mode='local'
                                )
                                print(f"PHASE 3: Local edit at step {step}")
                        else:
                            # Non-hybrid behavior uses configured editing mode
                            edit_mode = config.get("editing", {}).get("mode", "local")
                            edited_text = generate_edit(
                                example["text"], 
                                target_label=target_label,
                                mode=edit_mode,
                                current_task=task_name,
                                previous_tasks=previous_tasks
                            )

                        # PHASE 2: SEAL-enabled replay
                        # If SEAL mode is active compute real utility using
                        # score_edit_simple and include prediction/confidence
                        # information in the edit record saved to memory.
                        # In SEAL and Hybrid modes compute utility + pred/conf info
                        if seal_mode or hybrid_mode:
                            try:
                                pred_before, conf_before = trainer.predict_with_confidence(example["text"])
                            except Exception:
                                pred_before, conf_before = None, 0.0
                            try:
                                pred_after, conf_after = trainer.predict_with_confidence(edited_text)
                            except Exception:
                                pred_after, conf_after = None, 0.0

                            edit_record = {
                                "text": edited_text,
                                "label": target_label,
                                "task": task_name,
                                "original_text": example["text"],
                                "original_label": example["label"],
                                "pred_before": pred_before,
                                "pred_after": pred_after,
                                "conf_before": conf_before,
                                "conf_after": conf_after
                            }

                            # Compute utility using existing utility function
                            try:
                                utility = score_edit_simple(edit_record)
                            except Exception:
                                utility = 0.0
                            edit_record["utility"] = float(utility)
                            edited_batch.append(edit_record)
                        else:
                            # Baseline / non-SEAL behavior: keep original edit format
                            edited_batch.append({
                                "text": edited_text,
                                "label": target_label,
                                "task": task_name,
                                "original_text": example["text"],
                                "original_label": example["label"]
                            })
                    else:
                        # For other tasks, just use the original example for now
                        edited_batch.append({
                            "text": example["text"],
                            "label": example["label"],
                            "task": task_name
                        })
                except Exception as e:
                    print(f"⚠️ Error generating edit: {str(e)}")
            
            # In baseline mode we DO NOT sample from memory and DO NOT include
            # replay items in training. This ensures training batches contain
            # only current-task examples.
            replay_batch = []
            if not baseline_mode and memory and step > 0:  # Wait until we have some memory
                # Enforce light replay fraction between 0.1 and 0.2
                rf = config.get("replay", {}).get("fraction", config.get("replay", {}).get("frac", 0.15))
                try:
                    replay_fraction = float(rf)
                except Exception:
                    replay_fraction = 0.15
                if replay_fraction < 0.1 or replay_fraction > 0.2:
                    # Enforce default light replay
                    replay_fraction = 0.15
                replay_size = int(len(edited_batch) * replay_fraction)
                # Ensure at least 1 replay sample only if edited_batch is non-empty
                if replay_size > 0:
                    if hybrid_mode:
                        # Log light-replay settings per batch for Phase 3
                        print(f"PHASE 3: HYBRID SEAL - replay_fraction={replay_fraction:.2f}, replay_size={replay_size}")
                    # Get raw replay items. Use task-aware weighting when in hybrid mode.
                    if hybrid_mode and config.get("replay", {}).get("policy", "priority") == "priority":
                        edits_all = memory._read_all()
                        if len(edits_all) <= replay_size:
                            raw_replay = edits_all.copy()
                        else:
                            # Task-age weights
                            task_weights = {"imdb": 1.5, "squad": 1.2, "arc": 1.0}
                            eps = 1e-6
                            weights = []
                            for e in edits_all:
                                util = max(float(e.get("utility", 0.0)), 0.0)
                                task = str(e.get("task", "")).lower()
                                taw = task_weights.get(task, 1.0)
                                weights.append(util * taw + eps)

                            total = sum(weights)
                            if total <= 0:
                                raw_replay = memory.sample(
                                    replay_size,
                                    task_balance=config.get("replay", {}).get("task_balance", True),
                                    alpha=config.get("replay", {}).get("alpha", 1.0)
                                )
                            else:
                                import random as _random
                                indices = list(range(len(edits_all)))
                                w = weights.copy()
                                selected = []
                                for _ in range(replay_size):
                                    total_w = sum(w)
                                    if total_w <= 0:
                                        break
                                    probs = [x / total_w for x in w]
                                    idx = _random.choices(indices, weights=probs, k=1)[0]
                                    selected.append(edits_all[idx])
                                    w[idx] = 0.0
                                raw_replay = selected
                    else:
                        raw_replay = memory.sample(
                            replay_size,
                            task_balance=config.get("replay", {}).get("task_balance", True),
                            alpha=config.get("replay", {}).get("alpha", 1.0)
                        )

                    # Filter replay items to ensure label compatibility with current task
                    replay_batch = []
                    for r in raw_replay:
                        r_label = r.get("label")
                        # Accept sample if label is an int within the model's expected range
                        if isinstance(r_label, int) and r_label >= 0 and r_label < num_labels:
                            replay_batch.append({
                                "text": r.get("edit", r.get("original", "")),
                                "label": r_label,
                                "task": r.get("task", "unknown")
                            })
                        else:
                            continue

            # Combine batches and train (only using replay items compatible with current task)
            combined_batch = edited_batch + replay_batch
            random.shuffle(combined_batch)
            
            # Ensure we have valid data in the combined batch
            valid_batch = [item for item in combined_batch if "text" in item and "label" in item]
            if not valid_batch:
                print("⚠️ No valid examples in batch, skipping...")
                continue
                
            # Extract texts and labels
            texts = [item["text"] for item in valid_batch]
            labels = [item["label"] for item in valid_batch]
            
            # Train on the batch
            try:
                loss = trainer.train_on_batch(texts, labels)
            except Exception as e:
                print(f"⚠️ Error in training batch: {str(e)}")
                continue
            
            print(f"🔍 DEBUG: Considering {len(edited_batch)} edits for memory storage")
            # Store only high-utility edits in memory (PHASE 3: HYBRID SEAL)
            for edit in edited_batch:
                # Ensure there is a utility score; compute if missing
                if "utility" not in edit:
                    try:
                        edit_obj = {
                            "original": edit.get("original_text", edit.get("original", "")),
                            "edit": edit.get("text", ""),
                            "pred_before": edit.get("pred_before", ""),
                            "pred_after": edit.get("pred_after", ""),
                            "conf_before": edit.get("conf_before", 0.0),
                            "conf_after": edit.get("conf_after", 0.0)
                        }
                        edit["utility"] = float(score_edit_simple(edit_obj))
                    except Exception:
                        edit["utility"] = 0.0

                # Skip low-utility edits (threshold 0.3)
                try:
                    util = float(edit.get("utility", 0.0))
                except Exception:
                    util = 0.0

                if util < 0.3:
                    # Skip storing low-utility edits to keep memory minimal
                    if hybrid_mode:
                        print(f"PHASE 3: Skipping edit with utility={util:.3f}")
                    continue

                # Add to memory (preserve existing memory.add_edit behavior)
                memory.add_edit(edit)
            
            # Log progress
            if (step + 1) % 10 == 0 or step == steps_per_task - 1:
                print(f"📊 Task {task_name} | Step {step+1}/{steps_per_task} | Loss: {loss:.4f}")
        
        # Initialize task_metrics and task_evaluations for this task if they don't exist
        if task_name not in task_metrics:
            task_metrics[task_name] = []
        if task_name not in task_evaluations:
            task_evaluations[task_name] = []
            
        # Evaluate on all tasks seen so far
        print(f"\n🔍 Evaluating on all tasks after {task_name} (task {task_idx+1}/{len(task_order)})...")
        print(f"📊 Task metrics before evaluation: {task_metrics}")
        print(f"📊 Task evaluations before evaluation: {task_evaluations}")
        
        # Ensure we have a trainer instance with predict method
        if not hasattr(trainer, 'predict'):
            print("⚠️  Trainer does not have a predict method, creating a new one...")
            from seal.trainer import Trainer
            trainer = Trainer(
                model=model,
                tokenizer=tokenizer,
                learning_rate=config.get('learning_rate', 2e-5),
                device=config.get('device', 'cpu'),
                max_length=config.get('max_length', 512)
            )
            
        for eval_task in task_order[:task_idx+1]:
            print(f"\n  - Evaluating task: {eval_task}")
            try:
                val_set = task_val_sets.get(eval_task, [])
                print(f"  - Validation set size: {len(val_set)} examples")
                if not val_set:
                    print(f"⚠️  No validation set for task {eval_task}")
                    # If no validation set, use training data for evaluation
                    val_set = task_datasets.get(eval_task, [])
                    if not val_set:
                        print(f"⚠️  No data available for evaluation of task {eval_task}")
                        task_metrics[eval_task].append(None)
                        task_evaluations[eval_task].append(False)
                        continue
                    print(f"  - Using {len(val_set)} training examples for evaluation")
                
                # Limit evaluation to 100 examples for efficiency
                eval_size = min(100, len(val_set))
                val_set = val_set[:eval_size]
                texts = [ex["text"] for ex in val_set]
                true_labels = [ex["label"] for ex in val_set]
                
                try:
                    # Preserve original classifier and training state
                    original_clf = _copy.deepcopy(trainer.model.classifier)
                    original_training = trainer.model.training

                    # Explicitly attach the task-specific classifier head for evaluation
                    if eval_task in task_classifiers:
                        try:
                            # Create a fresh deepcopy of the stored classifier
                            task_clf = _copy.deepcopy(task_classifiers[eval_task])
                            # Explicitly move to the model's device and ensure it's on the same device
                            task_clf = task_clf.to(trainer.device)
                            trainer.model.classifier = task_clf
                            print(f"EVAL: Attached classifier head for task {eval_task}")
                        except Exception as e:
                            print(f"⚠️  EVAL: Failed to attach classifier for task {eval_task}: {e}")
                            # Fall back to original classifier if attachment fails
                            trainer.model.classifier = original_clf

                    # Ensure evaluation does not perform gradient updates
                    trainer.model.eval()
                    with torch.no_grad():
                        predictions = trainer.predict(texts)

                    # Restore original classifier and training state after evaluation
                    try:
                        trainer.model.classifier = original_clf.to(trainer.device)
                        print(f"EVAL: Restored classifier head after evaluating {eval_task}")
                    except Exception as e:
                        print(f"⚠️  EVAL: Failed to restore classifier after {eval_task}: {e}")
                    
                    # Restore original training state
                    if original_training:
                        trainer.model.train()

                    # Ensure labels and predictions are in the correct format
                    if not isinstance(true_labels, torch.Tensor):
                        true_labels = torch.tensor(true_labels, dtype=torch.long)

                    if isinstance(predictions, (list, np.ndarray)):
                        predictions = torch.tensor(predictions, dtype=torch.long)

                    # Calculate accuracy
                    correct = (predictions == true_labels).sum().item()
                    accuracy = correct / len(true_labels)

                    # Update metrics
                    task_metrics[eval_task].append(accuracy)
                    task_evaluations[eval_task].append(True)
                    print(f"   ✅ {eval_task.upper()} accuracy: {accuracy:.4f} ({correct}/{len(true_labels)})")

                except Exception as e:
                    print(f"   ⚠️  Error evaluating {eval_task}: {str(e)}")
                    task_metrics[eval_task].append(0.0)  # Default to 0 accuracy on error
                    task_evaluations[eval_task].append(False)
            
            except Exception as e:
                print(f"⚠️  Error in evaluation loop: {str(e)}")
                continue
        
        # Save results after each task
        # Save the task-specific classifier head for later evaluation/switching
        try:
            task_classifiers[task_name] = _copy.deepcopy(trainer.model.classifier)
        except Exception:
            pass

        # PHASE 5: Compute Fisher information after each task when EWC enabled
        if config.get('ewc', {}).get('enabled', False):
            try:
                val_set = task_val_sets.get(task_name, [])
                # Use validation set when available
                if not val_set:
                    val_set = task_datasets.get(task_name, [])
                trainer.compute_fisher(val_set, task_name)
                # Print the one-time PHASE 5 activation message after IMDB
                if task_idx == 0 and not hasattr(trainer, '_phase5_enabled'):
                    print("PHASE 5: EWC enabled (encoder-only) with task-specific classifier heads")
                    trainer._phase5_enabled = True
            except Exception as e:
                print(f"⚠️ Error computing Fisher for task {task_name}: {e}")
        # PHASE 4: After completing IMDB (task index 0) freeze low-level layers
        if task_idx == 0 and hybrid_mode:
            try:
                model_obj = trainer.model
                # Apply freezing to embeddings and first 4 transformer layers if present
                base = getattr(model_obj, 'distilbert', None)
                if base is not None:
                    # Freeze token embeddings
                    if hasattr(base, 'embeddings'):
                        for p in base.embeddings.parameters():
                            p.requires_grad = False
                    # Freeze first four transformer layers safely (layers 0..3)
                    if hasattr(base, 'transformer') and hasattr(base.transformer, 'layer'):
                        for i in range(4):
                            if i < len(base.transformer.layer):
                                for p in base.transformer.layer[i].parameters():
                                    p.requires_grad = False

                # Reinitialize optimizer so it only contains trainable params
                # Coerce config values to numeric types to avoid accidental string comparisons
                lr_raw = config.get('learning_rate', config.get('trainer', {}).get('learning_rate', 2e-5))
                try:
                    lr = float(lr_raw)
                except Exception:
                    lr = 2e-5
                wd_raw = config.get('weight_decay', config.get('training', {}).get('weight_decay', 0.01))
                try:
                    weight_decay = float(wd_raw)
                except Exception:
                    weight_decay = 0.01

                # Build concrete parameter list (avoid generator expressions that some optimizers may introspect)
                trainable_params = [p for p in trainer.model.parameters() if p.requires_grad]
                trainer.optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
                # One-time log for Phase 4.1
                if not hasattr(trainer, '_phase4_frozen'):
                    print("PHASE 4.1: Freezing embeddings and first 4 transformer layers after IMDB")
                    trainer._phase4_frozen = True
            except Exception as e:
                print(f"⚠️ Error while applying Phase 4 freezing: {e}")
    try:
        # Save results to outputs. Use separate directories for baseline and
        # SEAL-enabled runs to avoid clobbering artifacts.
        base_out = os.path.join(config.get("save_dir", "outputs"), "multi_task")
        if baseline_mode:
            base_out = os.path.join(base_out, "baseline")
        elif seal_mode:
            base_out = os.path.join(base_out, "seal")
        elif hybrid_mode:
            base_out = os.path.join(base_out, "hybrid")

        os.makedirs(base_out, exist_ok=True)
        results_file = os.path.join(base_out, "task_results.json")
        with open(results_file, "w") as f:
            import json
            json.dump(results, f, indent=2)
        print(f"💾 Saved evaluation results to {results_file}")
    except Exception as e:
        print(f"⚠️  Error saving results: {str(e)}")

    # Save task metrics after each task
    try:
        metrics_path = os.path.join(config.get("save_dir", "outputs"), "multi_task")
        if baseline_mode:
            metrics_path = os.path.join(metrics_path, "baseline")
        elif seal_mode:
            metrics_path = os.path.join(metrics_path, "seal")
        elif hybrid_mode:
            metrics_path = os.path.join(metrics_path, "hybrid")
        os.makedirs(metrics_path, exist_ok=True)
        metrics_file = os.path.join(metrics_path, f"{'_'.join(task_order[:task_idx+1])}_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump({"accuracy_matrix": task_metrics}, f, indent=2)
        print(f"💾 Saved task metrics to {metrics_file}")
    except Exception as e:
        print(f"⚠️  Error saving task metrics: {str(e)}")

    # Generate final evaluation report
    print("\n📊 Generating evaluation report...")
    from seal.eval_multi_domain import generate_evaluation_report
    
    # Prepare accuracy matrix
    accuracy_matrix = {}
    num_tasks = len(task_order)
    
    # Initialize the accuracy matrix with None values
    for task in task_order:
        accuracy_matrix[task] = [None] * num_tasks
    
    # Fill in the accuracies for each task at each evaluation point
    print("\n📊 Building accuracy matrix...")
    for eval_idx in range(num_tasks):
        current_tasks = task_order[:eval_idx+1]
        print(f"  - After task {eval_idx+1} (evaluating {len(current_tasks)} tasks)")
        for task in current_tasks:
            # Find the index of this task in the task order
            task_idx_in_order = task_order.index(task)
            # The task should have been evaluated (eval_idx - task_idx_in_order + 1) times
            # For example: imdb (idx 0) after task 2 (eval_idx=2) should have 3 metrics (indices 0,1,2)
            #              squad (idx 1) after task 2 (eval_idx=2) should have 2 metrics (indices 0,1)
            #              arc (idx 2) after task 2 (eval_idx=2) should have 1 metric (index 0)
            expected_metric_idx = eval_idx - task_idx_in_order
            
            # Check if we have enough metrics and if the evaluation was successful
            if (len(task_metrics[task]) > expected_metric_idx and 
                expected_metric_idx >= 0 and
                len(task_evaluations[task]) > expected_metric_idx and
                task_evaluations[task][expected_metric_idx]):
                accuracy_matrix[task][eval_idx] = task_metrics[task][expected_metric_idx]
                print(f"    - {task}: {accuracy_matrix[task][eval_idx]:.4f}")
            else:
                print(f"    - {task}: Not evaluated at this step (metrics len: {len(task_metrics[task])}, expected idx: {expected_metric_idx}, evaluations: {len(task_evaluations[task]) if task in task_evaluations else 0})")
    
    # Generate and save report
    # Generate evaluation report. Respect baseline output directory when in
    # baseline_mode so all Phase 1 artifacts are under outputs/multi_task/baseline/.
    report_out_dir = os.path.join(config.get("save_dir", "outputs"), "multi_task")
    if baseline_mode:
        report_out_dir = os.path.join(report_out_dir, "baseline")
    elif seal_mode:
        report_out_dir = os.path.join(report_out_dir, "seal")

    report = generate_evaluation_report(
        accuracy_matrix,
        output_dir=report_out_dir,
        prefix=f"{'_'.join(task_order)}_"
    )
    
    print("\n✅ Multi-task learning completed!")
    print(f"📝 Report saved to: {report_out_dir}")


if __name__ == "__main__":
    run_sequential_tasks()
