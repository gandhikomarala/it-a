# Unit Test for FailoverReadinessMetricExtractor_Modularnuclearsmr (Small Modular Nuclear Reactor (SMR) Systems).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.modular_nuclear_smr.failover_readiness_metric import FailoverReadinessMetricExtractor_Modularnuclearsmr
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_modular_nuclear_smr_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Modularnuclearsmr()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_modular_nuclear_smr_signal" in res.columns
    assert f"failover_readiness_metric_modular_nuclear_smr_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_modular_nuclear_smr_signal"].isnull().any()

def test_failover_readiness_metric_modular_nuclear_smr_empty():
    extractor = FailoverReadinessMetricExtractor_Modularnuclearsmr()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
