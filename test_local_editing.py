#!/usr/bin/env python
"""
SEAL Local Editing Metrics Test

This script evaluates the local editing model on a dataset and calculates various metrics.
"""

import os
import json
import time
import pickle
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from seal.adapter import generate_edit
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_metrics(results):
    """Calculate evaluation metrics from results."""
    original_texts = results['original_texts']
    edited_texts = results['edited_texts']
    true_labels = results['true_labels']
    pred_labels = results['pred_labels']
    
    print("\n📊 Calculating metrics...")
    
    # Calculate basic metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    avg_inference_time = np.mean(results['inference_times'])
    
    # Calculate BLEU and edit distance in batches to save memory
    batch_size = 1000
    bleu_scores = []
    edit_distances = []
    
    for i in tqdm(range(0, len(original_texts), batch_size), desc="Processing metrics"):
        batch_original = original_texts[i:i+batch_size]
        batch_edited = edited_texts[i:i+batch_size]
        
        # Calculate BLEU scores
        for orig, edit in zip(batch_original, batch_edited):
            try:
                orig_tokens = nltk.word_tokenize(orig.lower())
                edit_tokens = nltk.word_tokenize(edit.lower())
                smoothie = SmoothingFunction().method4
                score = sentence_bleu([orig_tokens], edit_tokens, smoothing_function=smoothie)
                bleu_scores.append(score)
            except:
                bleu_scores.append(0.0)
        
        # Calculate edit distances
        for orig, edit in zip(batch_original, batch_edited):
            try:
                dist = nltk.edit_distance(orig.split(), edit.split())
                edit_distances.append(dist)
            except:
                edit_distances.append(0)
    
    return {
        'accuracy': float(accuracy),
        'bleu_score': float(np.mean(bleu_scores)) if bleu_scores else 0.0,
        'avg_edit_distance': float(np.mean(edit_distances)) if edit_distances else 0.0,
        'avg_inference_time': float(avg_inference_time),
        'total_samples': len(original_texts)
    }

def test_local_editing(dataset_name="imdb", num_samples=None, batch_size=100, output_dir='outputs'):
    """Test local editing on a dataset with checkpointing."""
    print(f"🚀 Loading {dataset_name} dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'local_editing_metrics_{dataset_name}.json')
    checkpoint_file = os.path.join(output_dir, f'checkpoint_{dataset_name}.pkl')
    
    # Load dataset
    if dataset_name.lower() == "imdb":
        dataset = load_dataset("imdb", split='test')
        text_key = 'text'
        label_key = 'label'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Limit samples if specified
    if num_samples and num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Try to load checkpoint if exists
    start_idx = 0
    results = {
        'original_texts': [],
        'edited_texts': [],
        'true_labels': [],
        'pred_labels': [],
        'inference_times': []
    }
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                results = pickle.load(f)
            start_idx = len(results['original_texts'])
            print(f"📂 Loaded checkpoint with {start_idx} processed samples")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}. Starting from scratch.")
    
    print(f"🧪 Testing on {len(dataset)} samples (starting from {start_idx})...")
    
    # Process in batches
    for i in tqdm(range(start_idx, len(dataset)), desc="Processing"):
        item = dataset[i]
        text = item[text_key]
        label = item[label_key]
        
        try:
            # Generate edit
            start_time = time.time()
            edited_text = generate_edit(text, label=label)
            inference_time = time.time() - start_time
            
            # Store results
            results['original_texts'].append(text)
            results['edited_texts'].append(edited_text)
            results['true_labels'].append(label)
            results['pred_labels'].append(1 if 'positive' in edited_text.lower() else 0)
            results['inference_times'].append(inference_time)
            
            # Save checkpoint every batch_size samples
            if (i + 1) % batch_size == 0 or (i + 1) == len(dataset):
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(results, f)
                
        except Exception as e:
            print(f"\n⚠️  Error processing sample {i}: {str(e)}")
            continue
    
    # Remove checkpoint after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Prepare final results
    final_results = {
        'metrics': metrics,
        'config': {
            'dataset': dataset_name,
            'total_samples': len(dataset),
            'processed_samples': len(results['original_texts']),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'examples': [
            {
                'original': orig,
                'edited': edit,
                'true_label': int(tl),
                'pred_label': int(pl),
                'inference_time': float(time)
            }
            for orig, edit, tl, pl, time in zip(
                results['original_texts'][:10],  # First 10 examples
                results['edited_texts'][:10],
                results['true_labels'][:10],
                results['pred_labels'][:10],
                results['inference_times'][:10]
            )
        ]
    }
    
    # Save results
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n📊 Final Evaluation Results:")
    print(f"  • Processed Samples: {metrics['total_samples']:,}")
    print(f"  • Accuracy: {metrics['accuracy']:.4f}")
    print(f"  • BLEU Score: {metrics['bleu_score']:.4f}")
    print(f"  • Avg Edit Distance: {metrics['avg_edit_distance']:.2f}")
    print(f"  • Avg Inference Time: {metrics['avg_inference_time']:.4f}s")
    print(f"\n📝 Full results saved to: {os.path.abspath(results_file)}")
    
    return final_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test local editing metrics')
    parser.add_argument('--dataset', type=str, default='imdb',
                      help='Dataset to use for testing (default: imdb)')
    parser.add_argument('--num-samples', type=int, default=100,
                      help='Number of samples to test (default: 100)')
    
    args = parser.parse_args()
    
    test_local_editing(
        dataset_name=args.dataset,
        num_samples=args.num_samples
    )
