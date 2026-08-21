# Unit Test for EISNyquistSlopeExtractor_Autonomousunderwatergliders (Oceanographic Autonomous Underwater Gliders).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.autonomous_underwater_gliders.electrochemical_impedance_spectroscopy import EISNyquistSlopeExtractor_Autonomousunderwatergliders
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_electrochemical_impedance_spectroscopy_autonomous_underwater_gliders_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EISNyquistSlopeExtractor_Autonomousunderwatergliders()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"electrochemical_impedance_spectroscopy_autonomous_underwater_gliders_signal" in res.columns
    assert f"electrochemical_impedance_spectroscopy_autonomous_underwater_gliders_risk_score" in res.columns
    assert not res[f"electrochemical_impedance_spectroscopy_autonomous_underwater_gliders_signal"].isnull().any()

def test_electrochemical_impedance_spectroscopy_autonomous_underwater_gliders_empty():
    extractor = EISNyquistSlopeExtractor_Autonomousunderwatergliders()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
