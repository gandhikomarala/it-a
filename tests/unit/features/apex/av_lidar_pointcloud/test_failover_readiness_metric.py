# Unit Test for FailoverReadinessMetricExtractor_Avlidarpointcloud (Frequency Modulated Continuous Wave (FMCW) LiDAR).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.av_lidar_pointcloud.failover_readiness_metric import FailoverReadinessMetricExtractor_Avlidarpointcloud
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_av_lidar_pointcloud_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Avlidarpointcloud()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_av_lidar_pointcloud_signal" in res.columns
    assert f"failover_readiness_metric_av_lidar_pointcloud_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_av_lidar_pointcloud_signal"].isnull().any()

def test_failover_readiness_metric_av_lidar_pointcloud_empty():
    extractor = FailoverReadinessMetricExtractor_Avlidarpointcloud()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
