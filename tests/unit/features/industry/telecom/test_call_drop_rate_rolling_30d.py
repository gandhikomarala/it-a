# Unit Test for CallDropRate30d (telecom).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.telecom.call_drop_rate_rolling_30d import CallDropRate30d
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_call_drop_rate_rolling_30d_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CallDropRate30d()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"call_drop_rate_rolling_30d_signal" in res.columns
    assert f"call_drop_rate_rolling_30d_risk_index" in res.columns
    assert not res[f"call_drop_rate_rolling_30d_signal"].isnull().any()

def test_call_drop_rate_rolling_30d_empty_handling():
    extractor = CallDropRate30d()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
