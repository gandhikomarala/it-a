# Unit Test for FailoverReadinessMetricExtractor_Satellitesyntheticaperture (Spaceborne Synthetic Aperture Radar (SAR)).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.satellite_synthetic_aperture.failover_readiness_metric import FailoverReadinessMetricExtractor_Satellitesyntheticaperture
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_satellite_synthetic_aperture_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Satellitesyntheticaperture()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_satellite_synthetic_aperture_signal" in res.columns
    assert f"failover_readiness_metric_satellite_synthetic_aperture_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_satellite_synthetic_aperture_signal"].isnull().any()

def test_failover_readiness_metric_satellite_synthetic_aperture_empty():
    extractor = FailoverReadinessMetricExtractor_Satellitesyntheticaperture()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
