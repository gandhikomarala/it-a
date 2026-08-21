# Unit Test for DBConnectionPoolStarvationExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.database_connection_pool_starvation import DBConnectionPoolStarvationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_database_connection_pool_starvation_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DBConnectionPoolStarvationExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"database_connection_pool_starvation_signal" in res.columns
    assert f"database_connection_pool_starvation_risk_score" in res.columns
    assert not res[f"database_connection_pool_starvation_signal"].isnull().any()

def test_database_connection_pool_starvation_empty_dataframe():
    extractor = DBConnectionPoolStarvationExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
