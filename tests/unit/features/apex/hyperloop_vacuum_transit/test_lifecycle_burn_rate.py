# Unit Test for LifecycleBurnRateExtractor_Hyperloopvacuumtransit (Hyperloop Low-Pressure Pneumatic Transit).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hyperloop_vacuum_transit.lifecycle_burn_rate import LifecycleBurnRateExtractor_Hyperloopvacuumtransit
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_hyperloop_vacuum_transit_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Hyperloopvacuumtransit()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_hyperloop_vacuum_transit_signal" in res.columns
    assert f"lifecycle_burn_rate_hyperloop_vacuum_transit_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_hyperloop_vacuum_transit_signal"].isnull().any()

def test_lifecycle_burn_rate_hyperloop_vacuum_transit_empty():
    extractor = LifecycleBurnRateExtractor_Hyperloopvacuumtransit()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
