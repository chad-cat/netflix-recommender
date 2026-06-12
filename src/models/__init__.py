from .base import BaseRecommender
from .baseline import BaselineModel
from .item_cf import ItemBasedCF
from .user_cf import UserBasedCF
from .svd_mf import SVDModel

MODEL_REGISTRY = {
    "baseline": BaselineModel,
    "item_cf": ItemBasedCF,
    "user_cf": UserBasedCF,
    "svd": SVDModel,
}

__all__ = [
    "BaseRecommender",
    "BaselineModel",
    "ItemBasedCF",
    "UserBasedCF",
    "SVDModel",
    "MODEL_REGISTRY",
]
