# End-to-end preprocessing pipeline composer.
from typing import List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .imputers import AdaptiveImputer
from .outliers import OutlierHandler
from .transformers import FeatureInteractionTransformer
from .encoders import CategoricalEncoder
from .scalers import FeatureScaler
from packages.logging.logger import get_logger

logger = get_logger(__name__)

class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        scale_numeric: bool = True,
        handle_outliers: bool = True,
        drop_id_columns: bool = True
    ):
        self.scale_numeric = scale_numeric
        self.handle_outliers = handle_outliers
        self.drop_id_columns = drop_id_columns

        self.imputer = AdaptiveImputer()
        self.interaction_transformer = FeatureInteractionTransformer()
        self.outlier_handler = OutlierHandler() if handle_outliers else None
        self.encoder = CategoricalEncoder()
        self.scaler = FeatureScaler() if scale_numeric else None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None):
        X = pd.DataFrame(X).copy()
        id_cols = [c for c in ["customer_id", "email", "phone", "first_name", "last_name", "signup_date", "true_churn_probability"] if c in X.columns]
        if self.drop_id_columns and id_cols:
            X = X.drop(columns=id_cols)

        X = self.imputer.fit_transform(X)
        X = self.interaction_transformer.fit_transform(X)
        if self.outlier_handler:
            X = self.outlier_handler.fit_transform(X)
        if self.scaler:
            X = self.scaler.fit_transform(X)
        X = self.encoder.fit_transform(X)

        self.feature_names_ = X.columns.tolist()
        logger.info(f"Preprocessing pipeline fitted with {len(self.feature_names_)} engineered features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        id_cols = [c for c in ["customer_id", "email", "phone", "first_name", "last_name", "signup_date", "true_churn_probability"] if c in X.columns]
        if self.drop_id_columns and id_cols:
            X = X.drop(columns=id_cols)

        X = self.imputer.transform(X)
        X = self.interaction_transformer.transform(X)
        if self.outlier_handler:
            X = self.outlier_handler.transform(X)
        if self.scaler:
            X = self.scaler.transform(X)
        X = self.encoder.transform(X)

        X = X.reindex(columns=self.feature_names_, fill_value=0.0)
        return X
