# Comprehensive Unit Test for RANSpectrumUtilizationExtractor (Telecom 5G Network Slicing).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.telecom_5g_slicing.ran_spectrum_utilization_ratio import RANSpectrumUtilizationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ran_spectrum_utilization_ratio_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RANSpectrumUtilizationExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ran_spectrum_utilization_ratio_signal" in res.columns
    assert f"ran_spectrum_utilization_ratio_risk_score" in res.columns
    assert not res[f"ran_spectrum_utilization_ratio_signal"].isnull().any()

def test_ran_spectrum_utilization_ratio_empty_handling():
    extractor = RANSpectrumUtilizationExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
