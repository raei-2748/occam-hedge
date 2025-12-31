
import os
import random
import subprocess
import numpy as np

# Try to import torch, but don't fail if it's not installed yet, 
# although we should ensure it is installed.
try:
    import torch
except ImportError:
    torch = None

def set_seeds(seed: int = 42) -> None:
    """
    Enforce strict determinism across random, numpy, and torch (if available).
    
    Args:
        seed: The master seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Enforce strict determinism in cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_git_revision_hash() -> str:
    """
    Retrieve the current git commit hash.
    """
    try:
        # We assume the CWD is within the git repo or we can resolve relative to this file
        # But for safety, using the directory of this file is better.
        cwd = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL).decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "git-info-unavailable"
