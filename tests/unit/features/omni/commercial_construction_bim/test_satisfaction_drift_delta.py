# Unit Test for SatisfactionDriftDeltaExtractor_Commercialconstructionbim (Commercial BIM Construction Tracking).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.commercial_construction_bim.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Commercialconstructionbim
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_commercial_construction_bim_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Commercialconstructionbim()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_commercial_construction_bim_signal" in res.columns
    assert f"satisfaction_drift_delta_commercial_construction_bim_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_commercial_construction_bim_signal"].isnull().any()

def test_satisfaction_drift_delta_commercial_construction_bim_empty():
    extractor = SatisfactionDriftDeltaExtractor_Commercialconstructionbim()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
