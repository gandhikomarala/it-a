# Unit Test for FailoverReadinessMetricExtractor_Subsurfacegeophysicsseismic (3D Marine Seismic Geophysics Exploration).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.subsurface_geophysics_seismic.failover_readiness_metric import FailoverReadinessMetricExtractor_Subsurfacegeophysicsseismic
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_subsurface_geophysics_seismic_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Subsurfacegeophysicsseismic()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_subsurface_geophysics_seismic_signal" in res.columns
    assert f"failover_readiness_metric_subsurface_geophysics_seismic_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_subsurface_geophysics_seismic_signal"].isnull().any()

def test_failover_readiness_metric_subsurface_geophysics_seismic_empty():
    extractor = FailoverReadinessMetricExtractor_Subsurfacegeophysicsseismic()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
