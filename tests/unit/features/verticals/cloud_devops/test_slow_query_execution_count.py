# Unit Test for SlowQueryExecutionCountExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.slow_query_execution_count import SlowQueryExecutionCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_slow_query_execution_count_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SlowQueryExecutionCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"slow_query_execution_count_signal" in res.columns
    assert f"slow_query_execution_count_risk_score" in res.columns
    assert not res[f"slow_query_execution_count_signal"].isnull().any()

def test_slow_query_execution_count_empty_dataframe():
    extractor = SlowQueryExecutionCountExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
