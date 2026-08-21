# Unit Test for ComputeCPUSaturationExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.compute_cpu_saturation import ComputeCPUSaturationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_compute_cpu_saturation_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ComputeCPUSaturationExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"compute_cpu_saturation_signal" in res.columns
    assert f"compute_cpu_saturation_risk_score" in res.columns
    assert not res[f"compute_cpu_saturation_signal"].isnull().any()

def test_compute_cpu_saturation_empty_dataframe():
    extractor = ComputeCPUSaturationExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
