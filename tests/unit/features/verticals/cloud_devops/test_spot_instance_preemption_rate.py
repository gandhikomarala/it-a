# Unit Test for SpotInstancePreemptionRateExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.spot_instance_preemption_rate import SpotInstancePreemptionRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_spot_instance_preemption_rate_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SpotInstancePreemptionRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"spot_instance_preemption_rate_signal" in res.columns
    assert f"spot_instance_preemption_rate_risk_score" in res.columns
    assert not res[f"spot_instance_preemption_rate_signal"].isnull().any()

def test_spot_instance_preemption_rate_empty_dataframe():
    extractor = SpotInstancePreemptionRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
