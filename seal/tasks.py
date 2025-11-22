"""
Multi-domain task loaders for continual learning.

This module provides dataset loaders for different NLP tasks in a unified format.
"""
from typing import List, Dict, Any, Optional, Callable
from datasets import load_dataset
import random

def load_imdb(task_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load and format IMDB movie reviews dataset.
    
    Args:
        task_size: Number of examples to include (None for full dataset)
        
    Returns:
        List of examples with text, label, and task identifier
    """
    dataset = load_dataset("imdb", split="train")
    if task_size and task_size < len(dataset):
        dataset = dataset.select(range(task_size))
    
    examples = []
    for example in dataset:
        examples.append({
            "text": example["text"],
            "label": example["label"],
            "task": "imdb"
        })
    
    return examples

def load_squad(task_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load and format SQuAD v1.1 dataset.
    
    Args:
        task_size: Number of examples to include (None for full dataset)
        
    Returns:
        List of formatted QA examples with text, label, and task identifier
    """
    dataset = load_dataset("squad", split="train")
    if task_size and task_size < len(dataset):
        dataset = dataset.select(range(task_size))
    
    examples = []
    for example in dataset:
        formatted_text = (
            f"Context: {example['context']}\n"
            f"Question: {example['question']}\n"
            f"Answer: {example['answers']['text'][0]}"
        )
        examples.append({
            "text": formatted_text,
            "label": 1,  # Dummy label for compatibility
            "task": "squad"
        })
    
    return examples

def load_arc(task_size: Optional[int] = None, subset: str = "ARC-Easy") -> List[Dict[str, Any]]:
    """
    Load and format ARC (AI2 Reasoning Challenge) dataset.
    
    Args:
        task_size: Number of examples to include (None for full dataset)
        subset: Either 'ARC-Easy' or 'ARC-Challenge'
        
    Returns:
        List of formatted multiple-choice examples with text, label, and task identifier
    """
    dataset = load_dataset("ai2_arc", subset, split="train")
    if task_size and task_size < len(dataset):
        dataset = dataset.select(range(task_size))
    
    examples = []
    for example in dataset:
        choices = [f"{chr(65+i)}) {choice}" for i, choice in enumerate(example['choices']['text'])]
        formatted_text = (
            f"Problem: {example['question']}\n"
            f"Choices: {' '.join(choices)}\n"
            f"Answer: {example['answerKey']}"
        )
        
        # Convert answer key to label (0-3 for 4 choices)
        label = ord(example['answerKey']) - ord('A')
        
        examples.append({
            "text": formatted_text,
            "label": label,
            "task": "arc"
        })
    
    return examples

def get_task_loader(task_name: str) -> Callable[[Optional[int]], List[Dict[str, Any]]]:
    """
    Get the appropriate dataset loader function for a task.
    
    Args:
        task_name: Name of the task ('imdb', 'squad', or 'arc')
        
    Returns:
        A function that loads the specified dataset
        
    Raises:
        ValueError: If the task name is not recognized
    """
    loaders = {
        'imdb': load_imdb,
        'squad': load_squad,
        'arc': load_arc
    }
    
    if task_name not in loaders:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(loaders.keys())}")
    
    return loaders[task_name]
