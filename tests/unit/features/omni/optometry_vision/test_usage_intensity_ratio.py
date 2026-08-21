# Unit Test for UsageIntensityRatioExtractor_Optometryvision (Optometry & Optical Retail Chain).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.optometry_vision.usage_intensity_ratio import UsageIntensityRatioExtractor_Optometryvision
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_intensity_ratio_optometry_vision_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UsageIntensityRatioExtractor_Optometryvision()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"usage_intensity_ratio_optometry_vision_signal" in res.columns
    assert f"usage_intensity_ratio_optometry_vision_risk_score" in res.columns
    assert not res[f"usage_intensity_ratio_optometry_vision_signal"].isnull().any()

def test_usage_intensity_ratio_optometry_vision_empty():
    extractor = UsageIntensityRatioExtractor_Optometryvision()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
