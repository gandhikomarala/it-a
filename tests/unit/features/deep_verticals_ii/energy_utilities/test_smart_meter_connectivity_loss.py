# Comprehensive Unit Test for SmartMeterConnectivityLossExtractor (Energy & Smart Utilities).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.energy_utilities.smart_meter_connectivity_loss import SmartMeterConnectivityLossExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_smart_meter_connectivity_loss_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SmartMeterConnectivityLossExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"smart_meter_connectivity_loss_signal" in res.columns
    assert f"smart_meter_connectivity_loss_risk_score" in res.columns
    assert not res[f"smart_meter_connectivity_loss_signal"].isnull().any()

def test_smart_meter_connectivity_loss_empty_handling():
    extractor = SmartMeterConnectivityLossExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
