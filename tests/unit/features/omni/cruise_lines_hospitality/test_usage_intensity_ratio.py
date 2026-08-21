# Unit Test for UsageIntensityRatioExtractor_Cruiselineshospitality (Cruise Line Passenger Lifetime Value).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cruise_lines_hospitality.usage_intensity_ratio import UsageIntensityRatioExtractor_Cruiselineshospitality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_intensity_ratio_cruise_lines_hospitality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UsageIntensityRatioExtractor_Cruiselineshospitality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"usage_intensity_ratio_cruise_lines_hospitality_signal" in res.columns
    assert f"usage_intensity_ratio_cruise_lines_hospitality_risk_score" in res.columns
    assert not res[f"usage_intensity_ratio_cruise_lines_hospitality_signal"].isnull().any()

def test_usage_intensity_ratio_cruise_lines_hospitality_empty():
    extractor = UsageIntensityRatioExtractor_Cruiselineshospitality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
