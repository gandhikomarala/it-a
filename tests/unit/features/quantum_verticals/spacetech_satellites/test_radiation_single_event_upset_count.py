# Comprehensive Unit Test for RadiationSEUCountExtractor (SpaceTech & LEO Satellite Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.spacetech_satellites.radiation_single_event_upset_count import RadiationSEUCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_radiation_single_event_upset_count_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RadiationSEUCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"radiation_single_event_upset_count_signal" in res.columns
    assert f"radiation_single_event_upset_count_risk_score" in res.columns
    assert not res[f"radiation_single_event_upset_count_signal"].isnull().any()

def test_radiation_single_event_upset_count_empty_handling():
    extractor = RadiationSEUCountExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
