# Unit Test for RetentionHealthIndexExtractor_Civilinfrastructurebridges (Civil Infrastructure & Bridge Health).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.civil_infrastructure_bridges.retention_health_index import RetentionHealthIndexExtractor_Civilinfrastructurebridges
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_civil_infrastructure_bridges_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Civilinfrastructurebridges()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_civil_infrastructure_bridges_signal" in res.columns
    assert f"retention_health_index_civil_infrastructure_bridges_risk_score" in res.columns
    assert not res[f"retention_health_index_civil_infrastructure_bridges_signal"].isnull().any()

def test_retention_health_index_civil_infrastructure_bridges_empty():
    extractor = RetentionHealthIndexExtractor_Civilinfrastructurebridges()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
