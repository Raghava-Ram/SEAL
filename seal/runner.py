"""
SEAL Runner Module

Coordinates the SEAL self-adaptive learning loop, including model training and evaluation.
"""
import time
import os
import json
import random
import torch
from statistics import mean
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import yaml
from datasets import load_dataset
from torch import nn
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
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2  # Binary classification for IMDB
        )
        
        # Initialize trainer
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
                policy=config["replay"]["policy"]
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
    
    # Track task-specific data and metrics
    task_datasets = {}
    task_val_sets = {}
    task_metrics = {task: [] for task in task_order}
    
    # Load task datasets
    print("📥 Loading task datasets...")
    from seal.tasks import get_task_loader
    
    for task_name in task_order:
        try:
            loader = get_task_loader(task_name)
            task_data = loader()
            
            # Split into train/val
            if len(task_data) > eval_size * 2:
                task_datasets[task_name] = task_data[eval_size:]
                task_val_sets[task_name] = task_data[:eval_size]
            else:
                task_datasets[task_name] = task_data
                task_val_sets[task_name] = []
                
            print(f"   ✅ Loaded {len(task_datasets[task_name])} examples for {task_name}")
            
        except Exception as e:
            print(f"❌ Failed to load task {task_name}: {str(e)}")
            return
    
    # Main training loop over tasks
    for task_idx, task_name in enumerate(task_order):
        print(f"\n{'='*50}")
        print(f"🎯 Training on task {task_idx+1}/{len(task_order)}: {task_name.upper()}")
        print(f"{'='*50}")
        
        task_data = task_datasets[task_name]
        
        # Update model's classifier if needed (for tasks with different numbers of classes)
        num_labels = len(set(example["label"] for example in task_data))
        if hasattr(model, 'num_labels') and model.num_labels != num_labels:
            print(f"🔄 Updating classifier for {num_labels} classes")
            # Get the model configuration and update num_labels
            config = model.config
            config.num_labels = num_labels
            
            # Create a new model with the updated configuration
            device = model.device
            new_model = AutoModelForSequenceClassification.from_config(config)
            
            # Copy the weights from the old model to the new one
            # First, copy all the base model weights
            new_model.distilbert.load_state_dict(model.distilbert.state_dict())
            
            # Initialize the new classifier with the right dimensions
            in_features = model.classifier.in_features if hasattr(model, 'classifier') else model.config.hidden_size
            new_model.classifier = nn.Linear(in_features, num_labels)
            
            # If we're keeping the same device, move the new model there
            model = new_model.to(device)
            # Update the trainer's model reference
            trainer.model = model
        
        # Training loop for current task
        for step in range(steps_per_task):
            # Sample a batch
            batch = random.sample(task_data, min(config.get("batch_size", 32), len(task_data)))
            
            # Generate edits for the batch
            edited_batch = []
            for example in batch:
                try:
                    # For classification tasks, flip the label for editing
                    if task_name == "imdb":
                        target_label = 1 - example["label"]
                        edited_text = generate_edit(
                            example["text"], 
                            target_label=target_label,
                            mode=config.get("editing", {}).get("mode", "local")
                        )
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
            
            # Sample from memory for replay
            replay_batch = []
            if memory and step > 0:  # Wait until we have some memory
                replay_size = int(len(edited_batch) * config.get("replay", {}).get("fraction", 0.3))
                if replay_size > 0:
                    replay_batch = memory.sample(
                        replay_size,
                        task_balance=config.get("replay", {}).get("task_balance", True),
                        alpha=config.get("replay", {}).get("alpha", 1.0)
                    )
            
            # Combine batches and train
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
            
            # Store edits in memory with utility scores
            for edit in edited_batch:
                # Simple utility: 1.0 for now, can be enhanced
                edit["utility"] = 1.0
                memory.add_edit(edit)
            
            # Log progress
            if (step + 1) % 10 == 0 or step == steps_per_task - 1:
                print(f"📊 Task {task_name} | Step {step+1}/{steps_per_task} | Loss: {loss:.4f}")
        
        # Evaluate on all tasks seen so far
        print("\n🔍 Evaluating on all tasks...")
        for eval_task in task_order[:task_idx+1]:
            try:
                val_set = task_val_sets.get(eval_task, [])
                if not val_set:
                    print(f"⚠️  No validation set for task {eval_task}")
                    continue
                
                # Get texts and true labels
                texts = [ex["text"] for ex in val_set[:100]]  # Evaluate on first 100 examples
                true_labels = [ex["label"] for ex in val_set[:100]]
                
                # Get predictions in batches
                predictions = trainer.predict(texts)
                
                # Calculate accuracy
                accuracy = accuracy_score(true_labels, predictions)
                
                # Update results
                if eval_task not in results:
                    results[eval_task] = {}
                results[eval_task][f"after_task_{task_idx}"] = accuracy
                print(f"   ✅ {eval_task.upper()} accuracy: {accuracy:.4f}")
                
            except Exception as e:
                print(f"⚠️  Error evaluating {eval_task}: {str(e)}")
    
    # Generate final evaluation report
    print("\n📊 Generating evaluation report...")
    from seal.eval_multi_domain import generate_evaluation_report
    
    # Prepare accuracy matrix (fill with None for tasks not yet seen)
    accuracy_matrix = {}
    for i, task in enumerate(task_order):
        accs = task_metrics[task]
        # Pad with None for tasks not yet seen when this task was trained
        accs = [None] * i + accs
        accuracy_matrix[task] = accs
    
    # Generate and save report
    report = generate_evaluation_report(
        accuracy_matrix,
        output_dir=os.path.join(config.get("save_dir", "outputs"), "multi_task"),
        prefix=f"{'_'.join(task_order)}_"
    )
    
    print("\n✅ Multi-task learning completed!")
    print(f"📝 Report saved to: {os.path.join(config.get('save_dir', 'outputs'), 'multi_task')}")


if __name__ == "__main__":
    run_sequential_tasks()
