# Unit Test for FailoverReadinessMetricExtractor_Deepseaminingcrawlers (Abyssal Plain Polymetallic Nodule Harvesters).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.deepsea_mining_crawlers.failover_readiness_metric import FailoverReadinessMetricExtractor_Deepseaminingcrawlers
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_deepsea_mining_crawlers_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Deepseaminingcrawlers()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_deepsea_mining_crawlers_signal" in res.columns
    assert f"failover_readiness_metric_deepsea_mining_crawlers_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_deepsea_mining_crawlers_signal"].isnull().any()

def test_failover_readiness_metric_deepsea_mining_crawlers_empty():
    extractor = FailoverReadinessMetricExtractor_Deepseaminingcrawlers()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
