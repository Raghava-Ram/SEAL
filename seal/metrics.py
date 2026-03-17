from typing import List
import math


def _validate_matrix(acc_matrix: List[List[float]]):
    if not acc_matrix:
        raise ValueError("Accuracy matrix is empty")
    # ensure square matrix
    T = len(acc_matrix)
    for row in acc_matrix:
        if len(row) != T:
            raise ValueError("Accuracy matrix must be square (T x T). Found row length %d, expected %d" % (len(row), T))
        for v in row:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                raise ValueError("Accuracy matrix contains NaN or None values")


def final_average_accuracy(acc_matrix: List[List[float]]) -> float:
    """Return average of last row (final accuracies across tasks).

    Expects a square T x T accuracy matrix where row t contains accuracies
    of tasks up to t (upper triangular). The final row (index T-1) is used.
    """
    _validate_matrix(acc_matrix)
    last = acc_matrix[-1]
    return sum(last) / len(last)


def average_forgetting(acc_matrix: List[List[float]]) -> float:
    """Compute average forgetting across tasks.

    Using the definition: forgetting_i = max_{t<=T-1} acc_ti - acc_{T-1,i}
    and averaged across tasks (except possibly last diagonal if T==1).
    """
    _validate_matrix(acc_matrix)
    T = len(acc_matrix)
    forgetting = []
    for i in range(T):
        # historical max across all evaluations for task i (column i),
        # consider rows 0..T-1
        hist = [acc_matrix[t][i] for t in range(T)]
        historical_max = max(hist)
        final_acc = acc_matrix[-1][i]
        forgetting.append(max(0.0, historical_max - final_acc))
    if not forgetting:
        return 0.0
    return sum(forgetting) / len(forgetting)


def backward_transfer(acc_matrix: List[List[float]]) -> float:
    """Compute backward transfer (BWT).

    BWT = average over tasks i of (acc_{T-1,i} - acc_{i,i}).
    """
    _validate_matrix(acc_matrix)
    T = len(acc_matrix)
    bwt_vals = []
    for i in range(T):
        acc_after_all = acc_matrix[-1][i]
        acc_when_learned = acc_matrix[i][i]
        bwt_vals.append(acc_after_all - acc_when_learned)
    if not bwt_vals:
        return 0.0
    return sum(bwt_vals) / len(bwt_vals)
