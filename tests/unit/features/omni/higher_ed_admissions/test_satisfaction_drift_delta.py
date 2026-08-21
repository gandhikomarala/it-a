# Unit Test for SatisfactionDriftDeltaExtractor_Higheredadmissions (University Admissions & Enrollment).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.higher_ed_admissions.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Higheredadmissions
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_higher_ed_admissions_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Higheredadmissions()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_higher_ed_admissions_signal" in res.columns
    assert f"satisfaction_drift_delta_higher_ed_admissions_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_higher_ed_admissions_signal"].isnull().any()

def test_satisfaction_drift_delta_higher_ed_admissions_empty():
    extractor = SatisfactionDriftDeltaExtractor_Higheredadmissions()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
