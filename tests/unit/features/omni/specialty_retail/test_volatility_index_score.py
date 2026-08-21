# Unit Test for VolatilityIndexScoreExtractor_Specialtyretail (Specialty Retail Omnichannel Inventory).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.specialty_retail.volatility_index_score import VolatilityIndexScoreExtractor_Specialtyretail
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_specialty_retail_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Specialtyretail()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_specialty_retail_signal" in res.columns
    assert f"volatility_index_score_specialty_retail_risk_score" in res.columns
    assert not res[f"volatility_index_score_specialty_retail_signal"].isnull().any()

def test_volatility_index_score_specialty_retail_empty():
    extractor = VolatilityIndexScoreExtractor_Specialtyretail()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
