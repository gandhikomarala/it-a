# Comprehensive Unit Test for NelsonAalenHazardEstimator.
import pytest
import numpy as np
import pandas as pd
from ml.models.advanced.nelson_aalen_hazard import NelsonAalenHazardEstimator
from ml.data.synthetic_generator import SyntheticCustomerGenerator
from ml.preprocessing.pipeline import PreprocessingPipeline

def test_nelson_aalen_hazard_fit_and_predict():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(30)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    
    pipe = PreprocessingPipeline()
    X_trans = pipe.fit_transform(X)
    
    model = NelsonAalenHazardEstimator()
    model.fit(X_trans, y)
    probs = model.predict_proba(X_trans)
    preds = model.predict(X_trans)
    
    assert probs.shape == (30, 2)
    assert len(preds) == 30
    assert set(preds).issubset({0, 1})
