# Unit Test for IaCDriftExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.infrastructure_as_code_drift import IaCDriftExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_infrastructure_as_code_drift_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = IaCDriftExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"infrastructure_as_code_drift_signal" in res.columns
    assert f"infrastructure_as_code_drift_risk_score" in res.columns
    assert not res[f"infrastructure_as_code_drift_signal"].isnull().any()

def test_infrastructure_as_code_drift_empty_dataframe():
    extractor = IaCDriftExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
