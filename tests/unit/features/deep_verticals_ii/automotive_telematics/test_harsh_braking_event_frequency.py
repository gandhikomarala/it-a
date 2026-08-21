# Comprehensive Unit Test for HarshBrakingFrequencyExtractor (Automotive & Connected Fleet).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.automotive_telematics.harsh_braking_event_frequency import HarshBrakingFrequencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harsh_braking_event_frequency_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarshBrakingFrequencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harsh_braking_event_frequency_signal" in res.columns
    assert f"harsh_braking_event_frequency_risk_score" in res.columns
    assert not res[f"harsh_braking_event_frequency_signal"].isnull().any()

def test_harsh_braking_event_frequency_empty_handling():
    extractor = HarshBrakingFrequencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
