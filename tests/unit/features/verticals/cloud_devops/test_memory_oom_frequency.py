# Unit Test for MemoryOOMFrequencyExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.memory_oom_frequency import MemoryOOMFrequencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_memory_oom_frequency_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MemoryOOMFrequencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"memory_oom_frequency_signal" in res.columns
    assert f"memory_oom_frequency_risk_score" in res.columns
    assert not res[f"memory_oom_frequency_signal"].isnull().any()

def test_memory_oom_frequency_empty_dataframe():
    extractor = MemoryOOMFrequencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
