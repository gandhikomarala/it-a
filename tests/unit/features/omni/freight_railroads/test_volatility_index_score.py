# Unit Test for VolatilityIndexScoreExtractor_Freightrailroads (Class I Freight Railroad Logistics).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.freight_railroads.volatility_index_score import VolatilityIndexScoreExtractor_Freightrailroads
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_freight_railroads_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Freightrailroads()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_freight_railroads_signal" in res.columns
    assert f"volatility_index_score_freight_railroads_risk_score" in res.columns
    assert not res[f"volatility_index_score_freight_railroads_signal"].isnull().any()

def test_volatility_index_score_freight_railroads_empty():
    extractor = VolatilityIndexScoreExtractor_Freightrailroads()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
