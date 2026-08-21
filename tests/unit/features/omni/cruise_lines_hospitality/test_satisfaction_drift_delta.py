# Unit Test for SatisfactionDriftDeltaExtractor_Cruiselineshospitality (Cruise Line Passenger Lifetime Value).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cruise_lines_hospitality.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Cruiselineshospitality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_cruise_lines_hospitality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Cruiselineshospitality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_cruise_lines_hospitality_signal" in res.columns
    assert f"satisfaction_drift_delta_cruise_lines_hospitality_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_cruise_lines_hospitality_signal"].isnull().any()

def test_satisfaction_drift_delta_cruise_lines_hospitality_empty():
    extractor = SatisfactionDriftDeltaExtractor_Cruiselineshospitality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
