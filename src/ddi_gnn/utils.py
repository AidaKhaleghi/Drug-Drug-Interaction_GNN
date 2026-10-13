import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Returns the GPU if one is available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seeds Python, NumPy and PyTorch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_generator(generator: torch.Generator, num: int) -> list[torch.Generator]:
    """Creates `num` independent generators whose seeds come from `generator`."""
    seeds = torch.randint(0, 2**31 - 1, (num,), generator=generator)
    return [torch.Generator().manual_seed(int(seed)) for seed in seeds]
