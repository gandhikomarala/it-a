# Abstract base class for all machine learning model wrappers.
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

class BaseModelWrapper(ABC):
    def __init__(self, model_name: str, hyperparameters: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.hyperparameters = hyperparameters or {}
        self.model: Any = None
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "BaseModelWrapper":
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_feature_importances(self) -> Dict[str, float]:
        pass

    def get_params(self) -> Dict[str, Any]:
        return self.hyperparameters.copy()
