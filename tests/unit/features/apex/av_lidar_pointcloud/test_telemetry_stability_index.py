# Unit Test for TelemetryStabilityIndexExtractor_Avlidarpointcloud (Frequency Modulated Continuous Wave (FMCW) LiDAR).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.av_lidar_pointcloud.telemetry_stability_index import TelemetryStabilityIndexExtractor_Avlidarpointcloud
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_av_lidar_pointcloud_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Avlidarpointcloud()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_av_lidar_pointcloud_signal" in res.columns
    assert f"telemetry_stability_index_av_lidar_pointcloud_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_av_lidar_pointcloud_signal"].isnull().any()

def test_telemetry_stability_index_av_lidar_pointcloud_empty():
    extractor = TelemetryStabilityIndexExtractor_Avlidarpointcloud()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
