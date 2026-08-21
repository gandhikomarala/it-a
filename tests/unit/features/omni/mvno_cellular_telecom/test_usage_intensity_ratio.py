# Unit Test for UsageIntensityRatioExtractor_Mvnocellulartelecom (MVNO Mobile Virtual Network Operator).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.mvno_cellular_telecom.usage_intensity_ratio import UsageIntensityRatioExtractor_Mvnocellulartelecom
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_intensity_ratio_mvno_cellular_telecom_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UsageIntensityRatioExtractor_Mvnocellulartelecom()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"usage_intensity_ratio_mvno_cellular_telecom_signal" in res.columns
    assert f"usage_intensity_ratio_mvno_cellular_telecom_risk_score" in res.columns
    assert not res[f"usage_intensity_ratio_mvno_cellular_telecom_signal"].isnull().any()

def test_usage_intensity_ratio_mvno_cellular_telecom_empty():
    extractor = UsageIntensityRatioExtractor_Mvnocellulartelecom()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
