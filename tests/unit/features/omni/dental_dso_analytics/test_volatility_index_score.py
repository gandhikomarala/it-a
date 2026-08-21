# Unit Test for VolatilityIndexScoreExtractor_Dentaldsoanalytics (Dental DSO Practice Optimization).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.dental_dso_analytics.volatility_index_score import VolatilityIndexScoreExtractor_Dentaldsoanalytics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_dental_dso_analytics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Dentaldsoanalytics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_dental_dso_analytics_signal" in res.columns
    assert f"volatility_index_score_dental_dso_analytics_risk_score" in res.columns
    assert not res[f"volatility_index_score_dental_dso_analytics_signal"].isnull().any()

def test_volatility_index_score_dental_dso_analytics_empty():
    extractor = VolatilityIndexScoreExtractor_Dentaldsoanalytics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
