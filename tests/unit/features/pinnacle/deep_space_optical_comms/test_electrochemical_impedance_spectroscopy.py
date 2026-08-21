# Unit Test for EISNyquistSlopeExtractor_Deepspaceopticalcomms (Deep Space Optical Laser Communications).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.deep_space_optical_comms.electrochemical_impedance_spectroscopy import EISNyquistSlopeExtractor_Deepspaceopticalcomms
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_electrochemical_impedance_spectroscopy_deep_space_optical_comms_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EISNyquistSlopeExtractor_Deepspaceopticalcomms()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"electrochemical_impedance_spectroscopy_deep_space_optical_comms_signal" in res.columns
    assert f"electrochemical_impedance_spectroscopy_deep_space_optical_comms_risk_score" in res.columns
    assert not res[f"electrochemical_impedance_spectroscopy_deep_space_optical_comms_signal"].isnull().any()

def test_electrochemical_impedance_spectroscopy_deep_space_optical_comms_empty():
    extractor = EISNyquistSlopeExtractor_Deepspaceopticalcomms()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
