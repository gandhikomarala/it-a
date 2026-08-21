# Comprehensive Unit Test for AncillaryFeeSensitivityExtractor (Travel, Airline & Hospitality).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.travel_hospitality.ancillary_baggage_fee_sensitivity import AncillaryFeeSensitivityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ancillary_baggage_fee_sensitivity_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AncillaryFeeSensitivityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ancillary_baggage_fee_sensitivity_signal" in res.columns
    assert f"ancillary_baggage_fee_sensitivity_risk_score" in res.columns
    assert not res[f"ancillary_baggage_fee_sensitivity_signal"].isnull().any()

def test_ancillary_baggage_fee_sensitivity_empty_handling():
    extractor = AncillaryFeeSensitivityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
