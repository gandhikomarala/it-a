# Unit test for CoxProportionalHazardEstimator.
import pytest
import numpy as np
import pandas as pd
from ml.models.survival import CoxProportionalHazardEstimator
from ml.data.synthetic_generator import SyntheticCustomerGenerator
from ml.preprocessing.pipeline import PreprocessingPipeline

def test_survival_cox_train_predict():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(60)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    
    pipeline = PreprocessingPipeline()
    X_trans = pipeline.fit_transform(X)
    
    if "CoxProportionalHazardEstimator" == "CoxProportionalHazardEstimator":
        model = CoxProportionalHazardEstimator()
        model.fit(X_trans, df["tenure_months"], y)
        hazards = model.predict_hazard_ratio(X_trans)
        assert len(hazards) == 60
    else:
        model = CoxProportionalHazardEstimator()
        model.fit(X_trans, y)
        probs = model.predict_proba(X_trans)
        assert probs.shape == (60, 2)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
