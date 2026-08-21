# Unit Test for LifecycleBurnRateExtractor_Smartgridsynchrophasor (Smart Grid PMU Synchrophasor Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.smart_grid_synchrophasor.lifecycle_burn_rate import LifecycleBurnRateExtractor_Smartgridsynchrophasor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_smart_grid_synchrophasor_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Smartgridsynchrophasor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_smart_grid_synchrophasor_signal" in res.columns
    assert f"lifecycle_burn_rate_smart_grid_synchrophasor_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_smart_grid_synchrophasor_signal"].isnull().any()

def test_lifecycle_burn_rate_smart_grid_synchrophasor_empty():
    extractor = LifecycleBurnRateExtractor_Smartgridsynchrophasor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
