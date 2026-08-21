# Comprehensive Unit Test for AwardSeatAvailabilityExtractor (Travel, Airline & Hospitality).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.travel_hospitality.loyalty_award_seat_availability import AwardSeatAvailabilityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_loyalty_award_seat_availability_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AwardSeatAvailabilityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"loyalty_award_seat_availability_signal" in res.columns
    assert f"loyalty_award_seat_availability_risk_score" in res.columns
    assert not res[f"loyalty_award_seat_availability_signal"].isnull().any()

def test_loyalty_award_seat_availability_empty_handling():
    extractor = AwardSeatAvailabilityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
