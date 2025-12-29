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
    Load and format SQuAD v1.1 dataset for binary classification.
    
    This converts the QA task into a binary classification task where the model
    predicts whether a given answer is correct for a question-context pair.
    
    Args:
        task_size: Number of examples to include (None for full dataset)
        
    Returns:
        List of examples with text, label (1 for correct answer, 0 for incorrect),
        and task identifier
    """
    dataset = load_dataset("squad", split="train")
    if task_size and task_size < len(dataset):
        dataset = dataset.select(range(task_size // 2))  # We'll generate 2 examples per QA pair
    
    examples = []
    
    for example in dataset:
        context = example['context']
        question = example['question']
        correct_answer = example['answers']['text'][0]
        
        # Create positive example (correct answer)
        positive_text = f"Context: {context}\nQuestion: {question}\nAnswer: {correct_answer}"
        examples.append({
            "text": positive_text,
            "label": 1,  # Correct answer
            "task": "squad"
        })
        
        # Create negative example (incorrect answer)
        # Find another answer from a different example to use as negative
        neg_example = random.choice(dataset)
        while neg_example == example:
            neg_example = random.choice(dataset)
            
        negative_answer = neg_example['answers']['text'][0]
        negative_text = f"Context: {context}\nQuestion: {question}\nAnswer: {negative_answer}"
        
        examples.append({
            "text": negative_text,
            "label": 0,  # Incorrect answer
            "task": "squad"
        })
        
        # Early stopping if we've reached the desired number of examples
        if task_size and len(examples) >= task_size:
            break
    
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
        answer_key = example['answerKey']
        if isinstance(answer_key, str) and len(answer_key) > 0:
            # Handle letter-based answer keys (A, B, C, D, etc.)
            if answer_key[0].isalpha():
                label = ord(answer_key[0].upper()) - ord('A')
            # Handle numeric answer keys (0, 1, 2, 3, etc.)
            elif answer_key[0].isdigit():
                label = int(answer_key[0])
            else:
                # Skip invalid answer keys
                continue
        else:
            # Skip invalid answer keys
            continue
        
        # Ensure label is non-negative and within reasonable range (0-10 for safety)
        if label < 0 or label > 10:
            continue
        
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
