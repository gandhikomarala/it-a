# Unit Test for HarmonicDistortionFactorExtractor_Deepboreholegeothermalclosedloop (Supercritical Closed-Loop Deep Geothermal).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.deep_borehole_geothermal_closed_loop.harmonic_distortion_factor import HarmonicDistortionFactorExtractor_Deepboreholegeothermalclosedloop
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harmonic_distortion_factor_deep_borehole_geothermal_closed_loop_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarmonicDistortionFactorExtractor_Deepboreholegeothermalclosedloop()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harmonic_distortion_factor_deep_borehole_geothermal_closed_loop_signal" in res.columns
    assert f"harmonic_distortion_factor_deep_borehole_geothermal_closed_loop_risk_score" in res.columns
    assert not res[f"harmonic_distortion_factor_deep_borehole_geothermal_closed_loop_signal"].isnull().any()

def test_harmonic_distortion_factor_deep_borehole_geothermal_closed_loop_empty():
    extractor = HarmonicDistortionFactorExtractor_Deepboreholegeothermalclosedloop()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
