# Unit Test for ContractRenewalBarrierExtractor_Legalippatents (Patent Prosecution & IP Portfolio).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.legal_ip_patents.contract_renewal_barrier import ContractRenewalBarrierExtractor_Legalippatents
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_legal_ip_patents_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Legalippatents()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_legal_ip_patents_signal" in res.columns
    assert f"contract_renewal_barrier_legal_ip_patents_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_legal_ip_patents_signal"].isnull().any()

def test_contract_renewal_barrier_legal_ip_patents_empty():
    extractor = ContractRenewalBarrierExtractor_Legalippatents()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
