# Unit Test for ContractRenewalBarrierExtractor_Behavioralmentalhealth (Behavioral & Mental Health Telehealth).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.behavioral_mental_health.contract_renewal_barrier import ContractRenewalBarrierExtractor_Behavioralmentalhealth
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_behavioral_mental_health_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Behavioralmentalhealth()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_behavioral_mental_health_signal" in res.columns
    assert f"contract_renewal_barrier_behavioral_mental_health_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_behavioral_mental_health_signal"].isnull().any()

def test_contract_renewal_barrier_behavioral_mental_health_empty():
    extractor = ContractRenewalBarrierExtractor_Behavioralmentalhealth()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
