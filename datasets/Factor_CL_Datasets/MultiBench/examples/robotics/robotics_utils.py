import random

import numpy as np
import torch


def set_seeds(seed, use_cuda):
    """Set Seeds

    Args:
        seed (int): Sets the seed for numpy, torch and random
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
