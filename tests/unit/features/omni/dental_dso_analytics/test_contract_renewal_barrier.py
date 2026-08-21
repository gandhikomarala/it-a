# Unit Test for ContractRenewalBarrierExtractor_Dentaldsoanalytics (Dental DSO Practice Optimization).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.dental_dso_analytics.contract_renewal_barrier import ContractRenewalBarrierExtractor_Dentaldsoanalytics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_dental_dso_analytics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Dentaldsoanalytics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_dental_dso_analytics_signal" in res.columns
    assert f"contract_renewal_barrier_dental_dso_analytics_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_dental_dso_analytics_signal"].isnull().any()

def test_contract_renewal_barrier_dental_dso_analytics_empty():
    extractor = ContractRenewalBarrierExtractor_Dentaldsoanalytics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
