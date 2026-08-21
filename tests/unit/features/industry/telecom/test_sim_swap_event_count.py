# Unit Test for SIMSwapEventCount (telecom).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.telecom.sim_swap_event_count import SIMSwapEventCount
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_sim_swap_event_count_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SIMSwapEventCount()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"sim_swap_event_count_signal" in res.columns
    assert f"sim_swap_event_count_risk_index" in res.columns
    assert not res[f"sim_swap_event_count_signal"].isnull().any()

def test_sim_swap_event_count_empty_handling():
    extractor = SIMSwapEventCount()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
