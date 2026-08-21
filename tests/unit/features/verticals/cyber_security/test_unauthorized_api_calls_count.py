# Unit Test for UnauthorizedAPICallsExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.unauthorized_api_calls_count import UnauthorizedAPICallsExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_unauthorized_api_calls_count_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UnauthorizedAPICallsExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"unauthorized_api_calls_count_signal" in res.columns
    assert f"unauthorized_api_calls_count_risk_score" in res.columns
    assert not res[f"unauthorized_api_calls_count_signal"].isnull().any()

def test_unauthorized_api_calls_count_empty_dataframe():
    extractor = UnauthorizedAPICallsExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
