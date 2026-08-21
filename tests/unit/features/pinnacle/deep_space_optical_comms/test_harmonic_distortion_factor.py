# Unit Test for HarmonicDistortionFactorExtractor_Deepspaceopticalcomms (Deep Space Optical Laser Communications).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.deep_space_optical_comms.harmonic_distortion_factor import HarmonicDistortionFactorExtractor_Deepspaceopticalcomms
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harmonic_distortion_factor_deep_space_optical_comms_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarmonicDistortionFactorExtractor_Deepspaceopticalcomms()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harmonic_distortion_factor_deep_space_optical_comms_signal" in res.columns
    assert f"harmonic_distortion_factor_deep_space_optical_comms_risk_score" in res.columns
    assert not res[f"harmonic_distortion_factor_deep_space_optical_comms_signal"].isnull().any()

def test_harmonic_distortion_factor_deep_space_optical_comms_empty():
    extractor = HarmonicDistortionFactorExtractor_Deepspaceopticalcomms()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
