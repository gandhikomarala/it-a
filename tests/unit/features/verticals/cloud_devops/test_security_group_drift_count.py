# Unit Test for SecurityGroupDriftCountExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.security_group_drift_count import SecurityGroupDriftCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_security_group_drift_count_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SecurityGroupDriftCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"security_group_drift_count_signal" in res.columns
    assert f"security_group_drift_count_risk_score" in res.columns
    assert not res[f"security_group_drift_count_signal"].isnull().any()

def test_security_group_drift_count_empty_dataframe():
    extractor = SecurityGroupDriftCountExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
