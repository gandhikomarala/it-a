# Unit Test for AnomalousGeoLoginCountExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.anomalous_geo_login_count import AnomalousGeoLoginCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_anomalous_geo_login_count_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AnomalousGeoLoginCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"anomalous_geo_login_count_signal" in res.columns
    assert f"anomalous_geo_login_count_risk_score" in res.columns
    assert not res[f"anomalous_geo_login_count_signal"].isnull().any()

def test_anomalous_geo_login_count_empty_dataframe():
    extractor = AnomalousGeoLoginCountExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
