"""
SEAL Runner Module

Coordinates the SEAL self-adaptive learning loop, including model training and evaluation.
"""
import time
import os
import json
import matplotlib.pyplot as plt
from statistics import mean
from seal.adapter import generate_edit
from seal.trainer import SEALTrainer
from transformers import AutoTokenizer, AutoModelForMaskedLM
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
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        
        # Initialize trainer
        trainer = SEALTrainer(model, tokenizer, config)
        
        # Test text for the SEAL loop
        text = "Machine learning is transforming the world."
        
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
        for iteration in range(max_steps):
            print(f"\n🔁 SEAL Iteration {iteration+1}/{max_steps}")
            
            # 1. Generate edit
            edit = generate_edit(text, mode=config.get("edit_mode", "local"))
            print(f"✏️  Generated Edit: {edit}")
            
            # 2. Fine-tune model
            loss = trainer.train_step(edit)
            loss_history.append(loss)
            print(f"📉 Training Loss: {loss:.4f}")
            
            # 3. Periodically evaluate accuracy (every 2 steps or on last iteration)
            if iteration % 2 == 0 or iteration == max_steps - 1:
                acc = trainer.evaluate_accuracy(num_samples=100)
                accuracy_history.append(acc)
            
            # 4. Log system resources
            usage = trainer.get_system_usage()
            cpu_log.append(usage["cpu"])
            mem_log.append(usage["memory"])
            print(f"⚙️  CPU: {usage['cpu']}% | Memory: {usage['memory']}%")
            
            # Small delay for better readability
            time.sleep(1)
        
        # Save results
        results = {
            "loss": loss_history,
            "accuracy": accuracy_history,
            "cpu": cpu_log,
            "memory": mem_log,
            "avg_cpu": mean(cpu_log) if cpu_log else 0,
            "avg_memory": mean(mem_log) if mem_log else 0,
            "final_loss": loss_history[-1] if loss_history else None,
            "final_accuracy": accuracy_history[-1] if accuracy_history else None
        }
        
        # Save results to JSON
        with open("outputs/eval_results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        # Plot and save training loss
        plt.figure(figsize=(10, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(loss_history, 'b-', label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy if available
        if accuracy_history:
            plt.subplot(1, 2, 2)
            x = [i*2 for i in range(len(accuracy_history))]  # Map to correct iteration numbers
            plt.plot(x, accuracy_history, 'g-', label='Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig("outputs/training_metrics.png")
        
        print("\n✅ SEAL loop completed successfully!")
        print(f"📊 Results saved to outputs/ directory")
        print(f"   - Training metrics: outputs/training_metrics.png")
        print(f"   - Raw data: outputs/eval_results.json")
    
    except Exception as e:
        print(f"\n❌ Error during SEAL execution: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        return

    print("\n✅ SEAL architectural base loop executed successfully on CPU.")

if __name__ == "__main__":
    run_seal_loop()
