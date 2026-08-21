# Unit Test for LifecycleBurnRateExtractor_Directedenergyoptical (High-Energy Laser Beam Control Systems).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.directed_energy_optical.lifecycle_burn_rate import LifecycleBurnRateExtractor_Directedenergyoptical
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_directed_energy_optical_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Directedenergyoptical()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_directed_energy_optical_signal" in res.columns
    assert f"lifecycle_burn_rate_directed_energy_optical_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_directed_energy_optical_signal"].isnull().any()

def test_lifecycle_burn_rate_directed_energy_optical_empty():
    extractor = LifecycleBurnRateExtractor_Directedenergyoptical()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
