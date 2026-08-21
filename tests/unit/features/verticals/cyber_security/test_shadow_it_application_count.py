# Unit Test for ShadowITApplicationCountExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.shadow_it_application_count import ShadowITApplicationCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_shadow_it_application_count_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ShadowITApplicationCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"shadow_it_application_count_signal" in res.columns
    assert f"shadow_it_application_count_risk_score" in res.columns
    assert not res[f"shadow_it_application_count_signal"].isnull().any()

def test_shadow_it_application_count_empty_dataframe():
    extractor = ShadowITApplicationCountExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
