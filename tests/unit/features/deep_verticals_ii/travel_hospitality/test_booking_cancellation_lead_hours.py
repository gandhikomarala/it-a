# Comprehensive Unit Test for BookingCancellationLeadHoursExtractor (Travel, Airline & Hospitality).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.travel_hospitality.booking_cancellation_lead_hours import BookingCancellationLeadHoursExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_booking_cancellation_lead_hours_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BookingCancellationLeadHoursExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"booking_cancellation_lead_hours_signal" in res.columns
    assert f"booking_cancellation_lead_hours_risk_score" in res.columns
    assert not res[f"booking_cancellation_lead_hours_signal"].isnull().any()

def test_booking_cancellation_lead_hours_empty_handling():
    extractor = BookingCancellationLeadHoursExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
