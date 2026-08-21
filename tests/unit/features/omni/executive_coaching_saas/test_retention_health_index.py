# Unit Test for RetentionHealthIndexExtractor_Executivecoachingsaas (Executive Leadership Coaching SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.executive_coaching_saas.retention_health_index import RetentionHealthIndexExtractor_Executivecoachingsaas
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_executive_coaching_saas_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Executivecoachingsaas()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_executive_coaching_saas_signal" in res.columns
    assert f"retention_health_index_executive_coaching_saas_risk_score" in res.columns
    assert not res[f"retention_health_index_executive_coaching_saas_signal"].isnull().any()

def test_retention_health_index_executive_coaching_saas_empty():
    extractor = RetentionHealthIndexExtractor_Executivecoachingsaas()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
