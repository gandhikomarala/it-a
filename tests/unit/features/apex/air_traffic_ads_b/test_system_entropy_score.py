# Unit Test for SystemEntropyScoreExtractor_Airtrafficadsb (NextGen Air Traffic Control ADS-B Radar).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.air_traffic_ads_b.system_entropy_score import SystemEntropyScoreExtractor_Airtrafficadsb
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_air_traffic_ads_b_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Airtrafficadsb()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_air_traffic_ads_b_signal" in res.columns
    assert f"system_entropy_score_air_traffic_ads_b_risk_score" in res.columns
    assert not res[f"system_entropy_score_air_traffic_ads_b_signal"].isnull().any()

def test_system_entropy_score_air_traffic_ads_b_empty():
    extractor = SystemEntropyScoreExtractor_Airtrafficadsb()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
