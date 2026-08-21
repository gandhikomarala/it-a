# Unit Test for LifecycleBurnRateExtractor_Subseapipelinerov (Subsea Oil Pipeline & Deepsea ROV).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.subsea_pipeline_rov.lifecycle_burn_rate import LifecycleBurnRateExtractor_Subseapipelinerov
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_subsea_pipeline_rov_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Subseapipelinerov()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_subsea_pipeline_rov_signal" in res.columns
    assert f"lifecycle_burn_rate_subsea_pipeline_rov_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_subsea_pipeline_rov_signal"].isnull().any()

def test_lifecycle_burn_rate_subsea_pipeline_rov_empty():
    extractor = LifecycleBurnRateExtractor_Subseapipelinerov()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
