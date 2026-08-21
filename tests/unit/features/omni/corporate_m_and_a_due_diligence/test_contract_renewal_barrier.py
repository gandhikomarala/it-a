# Unit Test for ContractRenewalBarrierExtractor_Corporatemandaduediligence (Corporate M&A Virtual Data Room).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.corporate_m_and_a_due_diligence.contract_renewal_barrier import ContractRenewalBarrierExtractor_Corporatemandaduediligence
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_corporate_m_and_a_due_diligence_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Corporatemandaduediligence()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_corporate_m_and_a_due_diligence_signal" in res.columns
    assert f"contract_renewal_barrier_corporate_m_and_a_due_diligence_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_corporate_m_and_a_due_diligence_signal"].isnull().any()

def test_contract_renewal_barrier_corporate_m_and_a_due_diligence_empty():
    extractor = ContractRenewalBarrierExtractor_Corporatemandaduediligence()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
