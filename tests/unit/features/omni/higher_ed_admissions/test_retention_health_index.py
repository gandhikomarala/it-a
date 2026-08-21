# Unit Test for RetentionHealthIndexExtractor_Higheredadmissions (University Admissions & Enrollment).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.higher_ed_admissions.retention_health_index import RetentionHealthIndexExtractor_Higheredadmissions
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_higher_ed_admissions_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Higheredadmissions()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_higher_ed_admissions_signal" in res.columns
    assert f"retention_health_index_higher_ed_admissions_risk_score" in res.columns
    assert not res[f"retention_health_index_higher_ed_admissions_signal"].isnull().any()

def test_retention_health_index_higher_ed_admissions_empty():
    extractor = RetentionHealthIndexExtractor_Higheredadmissions()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
