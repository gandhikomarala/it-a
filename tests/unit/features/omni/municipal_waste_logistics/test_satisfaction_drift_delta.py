# Unit Test for SatisfactionDriftDeltaExtractor_Municipalwastelogistics (Municipal Smart Waste Routing).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.municipal_waste_logistics.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Municipalwastelogistics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_municipal_waste_logistics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Municipalwastelogistics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_municipal_waste_logistics_signal" in res.columns
    assert f"satisfaction_drift_delta_municipal_waste_logistics_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_municipal_waste_logistics_signal"].isnull().any()

def test_satisfaction_drift_delta_municipal_waste_logistics_empty():
    extractor = SatisfactionDriftDeltaExtractor_Municipalwastelogistics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
