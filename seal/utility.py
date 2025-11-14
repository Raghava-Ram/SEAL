"""
Utility scoring for SEAL edits.
"""

from typing import Dict, Any
import math

def score_edit_simple(edit_obj: Dict[str, Any], 
                     w_conf=0.6, w_change=0.3, w_acc=0.1) -> float:
    """
    Score an edit based on confidence change, prediction change, and accuracy.
    
    Args:
        edit_obj: Dictionary containing edit information
        w_conf: Weight for confidence change term
        w_change: Weight for prediction change term
        w_acc: Weight for accuracy term
        
    Returns:
        Utility score between 0 and 1
    """
    conf_before = float(edit_obj.get("conf_before", 0.0))
    conf_after = float(edit_obj.get("conf_after", 0.0))
    pred_before = str(edit_obj.get("pred_before", "")).lower()
    pred_after = str(edit_obj.get("pred_after", "")).lower()
    acc_before = float(edit_obj.get("acc_before", 0.0))
    acc_after = float(edit_obj.get("acc_after", 0.0))

    # Calculate utility components
    delta_conf = conf_after - conf_before
    change_flag = 1.0 if pred_after != pred_before else 0.0
    acc_delta = acc_after - acc_before

    # Normalize components
    conf_term = (math.tanh(delta_conf) + 1) / 2  # Map to [0,1]
    acc_term = 1 / (1 + math.exp(-5 * acc_delta))  # Sigmoid scaling

    # Weighted combination
    score = (w_conf * conf_term + 
             w_change * change_flag + 
             w_acc * acc_term) / (w_conf + w_change + w_acc)
             
    return max(0.0, min(1.0, score))  # Clamp to [0,1]
