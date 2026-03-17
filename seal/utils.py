def set_global_seed(seed: int):
    """Set global random seeds for reproducible CPU-only experiments.

    This sets Python, NumPy and PyTorch seeds and enables deterministic
    algorithms where possible. Caller should also disable CUDA visibility
    (e.g. `os.environ['CUDA_VISIBLE_DEVICES']=''`) to force CPU-only runs.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If CUDA is present, still set the cuda seeds (no GPU will be used if env blocks it)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # Make operations deterministic where possible
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older PyTorch versions may not have this API
        pass
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    return
