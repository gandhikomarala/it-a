# Comprehensive Unit Test for StarTrackerBlindingExtractor (SpaceTech & LEO Satellite Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.spacetech_satellites.star_tracker_optical_blinding_rate import StarTrackerBlindingExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_star_tracker_optical_blinding_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = StarTrackerBlindingExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"star_tracker_optical_blinding_rate_signal" in res.columns
    assert f"star_tracker_optical_blinding_rate_risk_score" in res.columns
    assert not res[f"star_tracker_optical_blinding_rate_signal"].isnull().any()

def test_star_tracker_optical_blinding_rate_empty_handling():
    extractor = StarTrackerBlindingExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
