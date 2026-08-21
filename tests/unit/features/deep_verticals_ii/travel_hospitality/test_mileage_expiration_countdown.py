# Comprehensive Unit Test for MileageExpirationCountdownExtractor (Travel, Airline & Hospitality).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.travel_hospitality.mileage_expiration_countdown import MileageExpirationCountdownExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_mileage_expiration_countdown_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MileageExpirationCountdownExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"mileage_expiration_countdown_signal" in res.columns
    assert f"mileage_expiration_countdown_risk_score" in res.columns
    assert not res[f"mileage_expiration_countdown_signal"].isnull().any()

def test_mileage_expiration_countdown_empty_handling():
    extractor = MileageExpirationCountdownExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
