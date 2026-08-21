# Unit Test for LoadBalancer5xxRateExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.load_balancer_5xx_rate import LoadBalancer5xxRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_load_balancer_5xx_rate_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LoadBalancer5xxRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"load_balancer_5xx_rate_signal" in res.columns
    assert f"load_balancer_5xx_rate_risk_score" in res.columns
    assert not res[f"load_balancer_5xx_rate_signal"].isnull().any()

def test_load_balancer_5xx_rate_empty_dataframe():
    extractor = LoadBalancer5xxRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
