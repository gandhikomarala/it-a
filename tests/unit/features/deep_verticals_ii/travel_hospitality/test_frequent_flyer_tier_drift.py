# Comprehensive Unit Test for FrequentFlyerTierDriftExtractor (Travel, Airline & Hospitality).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.travel_hospitality.frequent_flyer_tier_drift import FrequentFlyerTierDriftExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_frequent_flyer_tier_drift_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FrequentFlyerTierDriftExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"frequent_flyer_tier_drift_signal" in res.columns
    assert f"frequent_flyer_tier_drift_risk_score" in res.columns
    assert not res[f"frequent_flyer_tier_drift_signal"].isnull().any()

def test_frequent_flyer_tier_drift_empty_handling():
    extractor = FrequentFlyerTierDriftExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
