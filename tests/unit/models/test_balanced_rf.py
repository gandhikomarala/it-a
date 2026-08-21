# Unit test for BalancedRandomForestChurnModel.
import pytest
import numpy as np
import pandas as pd
from ml.models.benchmarks.balanced_random_forest import BalancedRandomForestChurnModel
from ml.data.synthetic_generator import SyntheticCustomerGenerator
from ml.preprocessing.pipeline import PreprocessingPipeline

def test_balanced_rf_train_predict():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(60)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    
    pipeline = PreprocessingPipeline()
    X_trans = pipeline.fit_transform(X)
    
    if "BalancedRandomForestChurnModel" == "CoxProportionalHazardEstimator":
        model = BalancedRandomForestChurnModel()
        model.fit(X_trans, df["tenure_months"], y)
        hazards = model.predict_hazard_ratio(X_trans)
        assert len(hazards) == 60
    else:
        model = BalancedRandomForestChurnModel()
        model.fit(X_trans, y)
        probs = model.predict_proba(X_trans)
        assert probs.shape == (60, 2)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
