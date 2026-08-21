# Unit Test for PredictiveWearVelocityExtractor_Avlidarpointcloud (Frequency Modulated Continuous Wave (FMCW) LiDAR).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.av_lidar_pointcloud.predictive_wear_velocity import PredictiveWearVelocityExtractor_Avlidarpointcloud
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_av_lidar_pointcloud_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Avlidarpointcloud()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_av_lidar_pointcloud_signal" in res.columns
    assert f"predictive_wear_velocity_av_lidar_pointcloud_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_av_lidar_pointcloud_signal"].isnull().any()

def test_predictive_wear_velocity_av_lidar_pointcloud_empty():
    extractor = PredictiveWearVelocityExtractor_Avlidarpointcloud()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
