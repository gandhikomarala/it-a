# Comprehensive Unit Test for OutageDurationMinutesExtractor (Energy & Smart Utilities).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.energy_utilities.outage_duration_minutes_exposure import OutageDurationMinutesExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_outage_duration_minutes_exposure_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OutageDurationMinutesExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"outage_duration_minutes_exposure_signal" in res.columns
    assert f"outage_duration_minutes_exposure_risk_score" in res.columns
    assert not res[f"outage_duration_minutes_exposure_signal"].isnull().any()

def test_outage_duration_minutes_exposure_empty_handling():
    extractor = OutageDurationMinutesExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
