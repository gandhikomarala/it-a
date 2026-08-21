# Unit Test for KubernetesPodRestartRateExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.kubernetes_pod_restart_rate import KubernetesPodRestartRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_kubernetes_pod_restart_rate_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = KubernetesPodRestartRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"kubernetes_pod_restart_rate_signal" in res.columns
    assert f"kubernetes_pod_restart_rate_risk_score" in res.columns
    assert not res[f"kubernetes_pod_restart_rate_signal"].isnull().any()

def test_kubernetes_pod_restart_rate_empty_dataframe():
    extractor = KubernetesPodRestartRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
