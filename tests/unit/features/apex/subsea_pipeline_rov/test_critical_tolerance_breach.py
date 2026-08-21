# Unit Test for CriticalToleranceBreachExtractor_Subseapipelinerov (Subsea Oil Pipeline & Deepsea ROV).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.subsea_pipeline_rov.critical_tolerance_breach import CriticalToleranceBreachExtractor_Subseapipelinerov
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_subsea_pipeline_rov_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Subseapipelinerov()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_subsea_pipeline_rov_signal" in res.columns
    assert f"critical_tolerance_breach_subsea_pipeline_rov_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_subsea_pipeline_rov_signal"].isnull().any()

def test_critical_tolerance_breach_subsea_pipeline_rov_empty():
    extractor = CriticalToleranceBreachExtractor_Subseapipelinerov()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
