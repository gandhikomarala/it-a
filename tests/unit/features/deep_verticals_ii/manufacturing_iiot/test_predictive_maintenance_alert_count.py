# Comprehensive Unit Test for PdMAlertCountExtractor (Manufacturing & Industrial IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.manufacturing_iiot.predictive_maintenance_alert_count import PdMAlertCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_maintenance_alert_count_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PdMAlertCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_maintenance_alert_count_signal" in res.columns
    assert f"predictive_maintenance_alert_count_risk_score" in res.columns
    assert not res[f"predictive_maintenance_alert_count_signal"].isnull().any()

def test_predictive_maintenance_alert_count_empty_handling():
    extractor = PdMAlertCountExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
