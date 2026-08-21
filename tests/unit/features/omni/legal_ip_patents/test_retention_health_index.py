# Unit Test for RetentionHealthIndexExtractor_Legalippatents (Patent Prosecution & IP Portfolio).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.legal_ip_patents.retention_health_index import RetentionHealthIndexExtractor_Legalippatents
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_legal_ip_patents_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Legalippatents()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_legal_ip_patents_signal" in res.columns
    assert f"retention_health_index_legal_ip_patents_risk_score" in res.columns
    assert not res[f"retention_health_index_legal_ip_patents_signal"].isnull().any()

def test_retention_health_index_legal_ip_patents_empty():
    extractor = RetentionHealthIndexExtractor_Legalippatents()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
