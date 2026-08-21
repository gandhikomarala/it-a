# Unit Test for SatisfactionDriftDeltaExtractor_Colocationdatacenter (Wholesale Colocation Datacenter Power).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.colocation_datacenter.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Colocationdatacenter
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_colocation_datacenter_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Colocationdatacenter()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_colocation_datacenter_signal" in res.columns
    assert f"satisfaction_drift_delta_colocation_datacenter_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_colocation_datacenter_signal"].isnull().any()

def test_satisfaction_drift_delta_colocation_datacenter_empty():
    extractor = SatisfactionDriftDeltaExtractor_Colocationdatacenter()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
