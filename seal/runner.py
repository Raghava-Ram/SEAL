"""
SEAL Runner Module

Coordinates the SEAL self-adaptive learning loop, including model training and evaluation.
"""
import time
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .trainer import SEALTrainer
from .adapter import generate_edit

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
        test_text = "Machine learning is transforming the world."
        
        # Run SEAL loop
        max_iterations = config.get("training", {}).get("max_steps", 3)
        
        for iteration in range(max_iterations):
            print(f"\n🔁 SEAL Iteration {iteration + 1}/{max_iterations}")
            
            # 1. Generate edit
            print("  🔄 Generating edit...")
            edit = generate_edit(test_text, mode=edit_mode)
            print(f"  ✏️  Generated Edit: {edit}")
            
            # 2. Fine-tune model
            print("  🏋️  Fine-tuning model...")
            loss = trainer.train_step(edit)
            print(f"  📉 Training loss: {loss:.4f}")
            
            # 3. Evaluate
            print("  🔍 Evaluating model...")
            outputs = trainer.evaluate(test_text)
            print("  ✅ Evaluation complete")
            
            # Small delay for better readability
            time.sleep(0.5)
            
    except Exception as e:
        print(f"\n❌ Error during SEAL execution: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        return

    print("\n✅ SEAL architectural base loop executed successfully on CPU.")

if __name__ == "__main__":
    run_seal_loop()
