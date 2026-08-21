# Unit Test for VolatilityIndexScoreExtractor_Hydroelectricdamiot (Hydroelectric Dam Structural Health).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.hydroelectric_dam_iot.volatility_index_score import VolatilityIndexScoreExtractor_Hydroelectricdamiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_hydroelectric_dam_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Hydroelectricdamiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_hydroelectric_dam_iot_signal" in res.columns
    assert f"volatility_index_score_hydroelectric_dam_iot_risk_score" in res.columns
    assert not res[f"volatility_index_score_hydroelectric_dam_iot_signal"].isnull().any()

def test_volatility_index_score_hydroelectric_dam_iot_empty():
    extractor = VolatilityIndexScoreExtractor_Hydroelectricdamiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
