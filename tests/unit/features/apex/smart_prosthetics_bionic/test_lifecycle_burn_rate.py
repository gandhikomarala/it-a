# Unit Test for LifecycleBurnRateExtractor_Smartprostheticsbionic (Myoelectric Bionic Prosthetics Control).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.smart_prosthetics_bionic.lifecycle_burn_rate import LifecycleBurnRateExtractor_Smartprostheticsbionic
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_smart_prosthetics_bionic_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Smartprostheticsbionic()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_smart_prosthetics_bionic_signal" in res.columns
    assert f"lifecycle_burn_rate_smart_prosthetics_bionic_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_smart_prosthetics_bionic_signal"].isnull().any()

def test_lifecycle_burn_rate_smart_prosthetics_bionic_empty():
    extractor = LifecycleBurnRateExtractor_Smartprostheticsbionic()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
