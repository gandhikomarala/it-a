# Comprehensive Unit Test for PeakKwhConsumptionShiftExtractor (Energy & Smart Utilities).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.energy_utilities.peak_kwh_consumption_shift import PeakKwhConsumptionShiftExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_peak_kwh_consumption_shift_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PeakKwhConsumptionShiftExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"peak_kwh_consumption_shift_signal" in res.columns
    assert f"peak_kwh_consumption_shift_risk_score" in res.columns
    assert not res[f"peak_kwh_consumption_shift_signal"].isnull().any()

def test_peak_kwh_consumption_shift_empty_handling():
    extractor = PeakKwhConsumptionShiftExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
