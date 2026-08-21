# Unit Test for LifecycleBurnRateExtractor_Opticaltransceiver800G (800G Silicon Photonics Optical Transceivers).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.optical_transceiver_800g.lifecycle_burn_rate import LifecycleBurnRateExtractor_Opticaltransceiver800G
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_optical_transceiver_800g_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Opticaltransceiver800G()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_optical_transceiver_800g_signal" in res.columns
    assert f"lifecycle_burn_rate_optical_transceiver_800g_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_optical_transceiver_800g_signal"].isnull().any()

def test_lifecycle_burn_rate_optical_transceiver_800g_empty():
    extractor = LifecycleBurnRateExtractor_Opticaltransceiver800G()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
