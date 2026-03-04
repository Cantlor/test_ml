"""Loss functions for multitask training."""

from net_train.losses.bwbl_loss import boundary_bwbl_loss
from net_train.losses.extent_loss import extent_loss

__all__ = [
    "boundary_bwbl_loss",
    "extent_loss",
]
