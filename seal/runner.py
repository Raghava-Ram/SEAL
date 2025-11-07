"""
SEAL Runner Module

Coordinates the SEAL self-adaptive learning loop, including model training and evaluation.
"""
import time
import os
import json
import random
import matplotlib.pyplot as plt
from statistics import mean
from seal.adapter import generate_edit
from seal.trainer import SEALTrainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml

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
