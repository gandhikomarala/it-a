# Unit Test for SatisfactionDriftDeltaExtractor_Reinsurancecatastrophe (Catastrophe Reinsurance Modeling).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.reinsurance_catastrophe.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Reinsurancecatastrophe
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_reinsurance_catastrophe_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Reinsurancecatastrophe()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_reinsurance_catastrophe_signal" in res.columns
    assert f"satisfaction_drift_delta_reinsurance_catastrophe_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_reinsurance_catastrophe_signal"].isnull().any()

def test_satisfaction_drift_delta_reinsurance_catastrophe_empty():
    extractor = SatisfactionDriftDeltaExtractor_Reinsurancecatastrophe()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
