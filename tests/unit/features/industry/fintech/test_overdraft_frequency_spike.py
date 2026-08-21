# Unit Test for OverdraftFrequencySpike (fintech).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.fintech.overdraft_frequency_spike import OverdraftFrequencySpike
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_overdraft_frequency_spike_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OverdraftFrequencySpike()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"overdraft_frequency_spike_signal" in res.columns
    assert f"overdraft_frequency_spike_risk_index" in res.columns
    assert not res[f"overdraft_frequency_spike_signal"].isnull().any()

def test_overdraft_frequency_spike_empty_handling():
    extractor = OverdraftFrequencySpike()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
