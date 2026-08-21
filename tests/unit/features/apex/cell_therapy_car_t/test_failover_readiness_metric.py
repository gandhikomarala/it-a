# Unit Test for FailoverReadinessMetricExtractor_Celltherapycart (Autologous CAR-T Cell Therapy Manufacturing).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.cell_therapy_car_t.failover_readiness_metric import FailoverReadinessMetricExtractor_Celltherapycart
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_cell_therapy_car_t_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Celltherapycart()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_cell_therapy_car_t_signal" in res.columns
    assert f"failover_readiness_metric_cell_therapy_car_t_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_cell_therapy_car_t_signal"].isnull().any()

def test_failover_readiness_metric_cell_therapy_car_t_empty():
    extractor = FailoverReadinessMetricExtractor_Celltherapycart()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
