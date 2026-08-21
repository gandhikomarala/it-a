# Unit Test for UsageIntensityRatioExtractor_Specialtyretail (Specialty Retail Omnichannel Inventory).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.specialty_retail.usage_intensity_ratio import UsageIntensityRatioExtractor_Specialtyretail
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_intensity_ratio_specialty_retail_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UsageIntensityRatioExtractor_Specialtyretail()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"usage_intensity_ratio_specialty_retail_signal" in res.columns
    assert f"usage_intensity_ratio_specialty_retail_risk_score" in res.columns
    assert not res[f"usage_intensity_ratio_specialty_retail_signal"].isnull().any()

def test_usage_intensity_ratio_specialty_retail_empty():
    extractor = UsageIntensityRatioExtractor_Specialtyretail()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
