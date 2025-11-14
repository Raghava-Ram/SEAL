"""
SEAL Runner Module

Coordinates the SEAL self-adaptive learning loop, including model training and evaluation.
"""
import time
import os
import json
import random
from statistics import mean
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from seal.llm_adapter import get_llm_client
from seal.prompt_adapter import create_prompt
from seal.trainer import SEALTrainer
from seal.adapter import generate_edit

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
                edited_text = generate_edit(sample["text"])
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
        
        # Run SEAL loop
        max_steps = config.get("max_steps", 10)
        
        # Initial evaluation before any training
        print("\n🔍 Running initial evaluation...")
        initial_acc = trainer.evaluate_accuracy(imdb_data[:100])
        accuracy_history.append(initial_acc)
        print(f"📊 Initial Accuracy: {initial_acc:.4f}")
        
        for iteration in range(max_steps):
            print(f"\n🔁 SEAL Iteration {iteration+1}/{max_steps}")
            
            # 1. Sample from IMDB and generate edit
            sample = random.choice(imdb_data)
            original_text = sample["text"][:200] + "..."  # Truncate for display
            edit = generate_edit(sample["text"], mode=config.get("edit_mode", "local"))
            print(f"📝 Original: {original_text}")
            print(f"✏️  Edited  : {edit[:200]}...")
            
            # 2. Fine-tune model on the edited text (use the sample's label)
            loss = trainer.train_step(edit, label=sample["label"])
            loss_history.append(loss)
            print(f"📉 Training Loss: {loss:.4f}")
            
            # 3. Periodically evaluate accuracy (every 2 steps or on last iteration)
            if (iteration + 1) % 2 == 0 or iteration == max_steps - 1:
                acc = trainer.evaluate_accuracy(imdb_data[:100])  # Use first 100 samples for evaluation
                accuracy_history.append(acc)
            
            # 4. Log system resources
            usage = trainer.get_system_usage()
            cpu_log.append(usage["cpu"])
            mem_log.append(usage["memory"])
            print(f"⚙️  CPU: {usage['cpu']}% | Memory: {usage['memory']}%")
            
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

if __name__ == "__main__":
    run_seal_loop()
