# Unit Test for UsageIntensityRatioExtractor_Luxuryhospitality (Luxury Resort Concierge Guest Experience).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.luxury_hospitality.usage_intensity_ratio import UsageIntensityRatioExtractor_Luxuryhospitality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_intensity_ratio_luxury_hospitality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UsageIntensityRatioExtractor_Luxuryhospitality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"usage_intensity_ratio_luxury_hospitality_signal" in res.columns
    assert f"usage_intensity_ratio_luxury_hospitality_risk_score" in res.columns
    assert not res[f"usage_intensity_ratio_luxury_hospitality_signal"].isnull().any()

def test_usage_intensity_ratio_luxury_hospitality_empty():
    extractor = UsageIntensityRatioExtractor_Luxuryhospitality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
