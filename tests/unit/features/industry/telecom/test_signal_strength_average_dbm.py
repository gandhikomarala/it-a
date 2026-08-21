# Unit Test for SignalStrengthAverageDbm (telecom).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.telecom.signal_strength_average_dbm import SignalStrengthAverageDbm
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_signal_strength_average_dbm_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SignalStrengthAverageDbm()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"signal_strength_average_dbm_signal" in res.columns
    assert f"signal_strength_average_dbm_risk_index" in res.columns
    assert not res[f"signal_strength_average_dbm_signal"].isnull().any()

def test_signal_strength_average_dbm_empty_handling():
    extractor = SignalStrengthAverageDbm()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
