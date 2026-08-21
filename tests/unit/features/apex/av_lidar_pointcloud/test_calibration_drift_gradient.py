# Unit Test for CalibrationDriftGradientExtractor_Avlidarpointcloud (Frequency Modulated Continuous Wave (FMCW) LiDAR).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.av_lidar_pointcloud.calibration_drift_gradient import CalibrationDriftGradientExtractor_Avlidarpointcloud
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_av_lidar_pointcloud_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Avlidarpointcloud()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_av_lidar_pointcloud_signal" in res.columns
    assert f"calibration_drift_gradient_av_lidar_pointcloud_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_av_lidar_pointcloud_signal"].isnull().any()

def test_calibration_drift_gradient_av_lidar_pointcloud_empty():
    extractor = CalibrationDriftGradientExtractor_Avlidarpointcloud()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
