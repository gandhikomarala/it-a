# Unit Test for RetentionHealthIndexExtractor_Corporatemandaduediligence (Corporate M&A Virtual Data Room).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.corporate_m_and_a_due_diligence.retention_health_index import RetentionHealthIndexExtractor_Corporatemandaduediligence
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_corporate_m_and_a_due_diligence_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Corporatemandaduediligence()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_corporate_m_and_a_due_diligence_signal" in res.columns
    assert f"retention_health_index_corporate_m_and_a_due_diligence_risk_score" in res.columns
    assert not res[f"retention_health_index_corporate_m_and_a_due_diligence_signal"].isnull().any()

def test_retention_health_index_corporate_m_and_a_due_diligence_empty():
    extractor = RetentionHealthIndexExtractor_Corporatemandaduediligence()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
