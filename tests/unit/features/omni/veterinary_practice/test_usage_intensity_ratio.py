# Unit Test for UsageIntensityRatioExtractor_Veterinarypractice (Veterinary Practice Management).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.veterinary_practice.usage_intensity_ratio import UsageIntensityRatioExtractor_Veterinarypractice
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_intensity_ratio_veterinary_practice_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UsageIntensityRatioExtractor_Veterinarypractice()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"usage_intensity_ratio_veterinary_practice_signal" in res.columns
    assert f"usage_intensity_ratio_veterinary_practice_risk_score" in res.columns
    assert not res[f"usage_intensity_ratio_veterinary_practice_signal"].isnull().any()

def test_usage_intensity_ratio_veterinary_practice_empty():
    extractor = UsageIntensityRatioExtractor_Veterinarypractice()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
