# Unit Test for SystemEntropyScoreExtractor_Geostationarycommsatpayload (GEO High-Throughput Satellite Spot Beams).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.geostationary_commsat_payload.system_entropy_score import SystemEntropyScoreExtractor_Geostationarycommsatpayload
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_geostationary_commsat_payload_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Geostationarycommsatpayload()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_geostationary_commsat_payload_signal" in res.columns
    assert f"system_entropy_score_geostationary_commsat_payload_risk_score" in res.columns
    assert not res[f"system_entropy_score_geostationary_commsat_payload_signal"].isnull().any()

def test_system_entropy_score_geostationary_commsat_payload_empty():
    extractor = SystemEntropyScoreExtractor_Geostationarycommsatpayload()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
