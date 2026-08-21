# Comprehensive Unit Test for TractorTelematicsIdleHoursExtractor (Agriculture & Precision Farming).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.agriculture_agtech.tractor_telematics_idle_hours import TractorTelematicsIdleHoursExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_tractor_telematics_idle_hours_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TractorTelematicsIdleHoursExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"tractor_telematics_idle_hours_signal" in res.columns
    assert f"tractor_telematics_idle_hours_risk_score" in res.columns
    assert not res[f"tractor_telematics_idle_hours_signal"].isnull().any()

def test_tractor_telematics_idle_hours_empty_handling():
    extractor = TractorTelematicsIdleHoursExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
