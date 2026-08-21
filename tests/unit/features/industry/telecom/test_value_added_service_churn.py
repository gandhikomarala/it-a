# Unit Test for ValueAddedServiceChurn (telecom).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.telecom.value_added_service_churn import ValueAddedServiceChurn
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_value_added_service_churn_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ValueAddedServiceChurn()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"value_added_service_churn_signal" in res.columns
    assert f"value_added_service_churn_risk_index" in res.columns
    assert not res[f"value_added_service_churn_signal"].isnull().any()

def test_value_added_service_churn_empty_handling():
    extractor = ValueAddedServiceChurn()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
