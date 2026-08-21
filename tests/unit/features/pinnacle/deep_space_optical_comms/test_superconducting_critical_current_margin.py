# Unit Test for CriticalCurrentMarginExtractor_Deepspaceopticalcomms (Deep Space Optical Laser Communications).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.deep_space_optical_comms.superconducting_critical_current_margin import CriticalCurrentMarginExtractor_Deepspaceopticalcomms
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_superconducting_critical_current_margin_deep_space_optical_comms_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalCurrentMarginExtractor_Deepspaceopticalcomms()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"superconducting_critical_current_margin_deep_space_optical_comms_signal" in res.columns
    assert f"superconducting_critical_current_margin_deep_space_optical_comms_risk_score" in res.columns
    assert not res[f"superconducting_critical_current_margin_deep_space_optical_comms_signal"].isnull().any()

def test_superconducting_critical_current_margin_deep_space_optical_comms_empty():
    extractor = CriticalCurrentMarginExtractor_Deepspaceopticalcomms()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
