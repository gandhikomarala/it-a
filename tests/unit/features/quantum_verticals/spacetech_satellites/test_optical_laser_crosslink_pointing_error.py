# Comprehensive Unit Test for OpticalLaserCrosslinkErrorExtractor (SpaceTech & LEO Satellite Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.spacetech_satellites.optical_laser_crosslink_pointing_error import OpticalLaserCrosslinkErrorExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_laser_crosslink_pointing_error_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OpticalLaserCrosslinkErrorExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_laser_crosslink_pointing_error_signal" in res.columns
    assert f"optical_laser_crosslink_pointing_error_risk_score" in res.columns
    assert not res[f"optical_laser_crosslink_pointing_error_signal"].isnull().any()

def test_optical_laser_crosslink_pointing_error_empty_handling():
    extractor = OpticalLaserCrosslinkErrorExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
