# Unit Test for LifecycleBurnRateExtractor_Geostationarycommsatpayload (GEO High-Throughput Satellite Spot Beams).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.geostationary_commsat_payload.lifecycle_burn_rate import LifecycleBurnRateExtractor_Geostationarycommsatpayload
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_geostationary_commsat_payload_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Geostationarycommsatpayload()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_geostationary_commsat_payload_signal" in res.columns
    assert f"lifecycle_burn_rate_geostationary_commsat_payload_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_geostationary_commsat_payload_signal"].isnull().any()

def test_lifecycle_burn_rate_geostationary_commsat_payload_empty():
    extractor = LifecycleBurnRateExtractor_Geostationarycommsatpayload()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
