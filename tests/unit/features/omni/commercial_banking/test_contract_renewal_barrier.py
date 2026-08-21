# Unit Test for ContractRenewalBarrierExtractor_Commercialbanking (Commercial Treasury & Syndicated Lending).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.commercial_banking.contract_renewal_barrier import ContractRenewalBarrierExtractor_Commercialbanking
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_commercial_banking_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Commercialbanking()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_commercial_banking_signal" in res.columns
    assert f"contract_renewal_barrier_commercial_banking_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_commercial_banking_signal"].isnull().any()

def test_contract_renewal_barrier_commercial_banking_empty():
    extractor = ContractRenewalBarrierExtractor_Commercialbanking()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
