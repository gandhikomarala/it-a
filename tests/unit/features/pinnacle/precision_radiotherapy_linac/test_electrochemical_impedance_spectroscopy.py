# Unit Test for EISNyquistSlopeExtractor_Precisionradiotherapylinac (Linear Accelerator Medical Radiotherapy).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.precision_radiotherapy_linac.electrochemical_impedance_spectroscopy import EISNyquistSlopeExtractor_Precisionradiotherapylinac
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_electrochemical_impedance_spectroscopy_precision_radiotherapy_linac_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EISNyquistSlopeExtractor_Precisionradiotherapylinac()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"electrochemical_impedance_spectroscopy_precision_radiotherapy_linac_signal" in res.columns
    assert f"electrochemical_impedance_spectroscopy_precision_radiotherapy_linac_risk_score" in res.columns
    assert not res[f"electrochemical_impedance_spectroscopy_precision_radiotherapy_linac_signal"].isnull().any()

def test_electrochemical_impedance_spectroscopy_precision_radiotherapy_linac_empty():
    extractor = EISNyquistSlopeExtractor_Precisionradiotherapylinac()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
