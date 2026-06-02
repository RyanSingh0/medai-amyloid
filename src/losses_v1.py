"""
Loss functions for Amyloid PET Centiloid Prediction.

Add custom loss classes here and register them in get_criterion().
Huber loss class for mse and mae with criterion. 
"""

import torch.nn as nn


def get_criterion(name: str = "huber", **kwargs):
    """Factory for loss functions.

    Args:
        name: One of "mse", "mae", "huber".
        **kwargs: Passed to the loss constructor.
    """
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "huber":
        return nn.HuberLoss(delta = 15.0)
    else:
        raise ValueError(f"Unknown loss: {name!r}. Choose from: mse, mae, huber")
