# Comprehensive Unit Test for FlightDelayClaimsExtractor (Travel, Airline & Hospitality).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.travel_hospitality.flight_delay_compensation_claims import FlightDelayClaimsExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_flight_delay_compensation_claims_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FlightDelayClaimsExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"flight_delay_compensation_claims_signal" in res.columns
    assert f"flight_delay_compensation_claims_risk_score" in res.columns
    assert not res[f"flight_delay_compensation_claims_signal"].isnull().any()

def test_flight_delay_compensation_claims_empty_handling():
    extractor = FlightDelayClaimsExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
