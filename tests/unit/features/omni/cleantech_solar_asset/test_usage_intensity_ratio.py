# Unit Test for UsageIntensityRatioExtractor_Cleantechsolarasset (Utility Solar Asset Performance).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cleantech_solar_asset.usage_intensity_ratio import UsageIntensityRatioExtractor_Cleantechsolarasset
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_intensity_ratio_cleantech_solar_asset_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UsageIntensityRatioExtractor_Cleantechsolarasset()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"usage_intensity_ratio_cleantech_solar_asset_signal" in res.columns
    assert f"usage_intensity_ratio_cleantech_solar_asset_risk_score" in res.columns
    assert not res[f"usage_intensity_ratio_cleantech_solar_asset_signal"].isnull().any()

def test_usage_intensity_ratio_cleantech_solar_asset_empty():
    extractor = UsageIntensityRatioExtractor_Cleantechsolarasset()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
