# Unit Test for ContractRenewalBarrierExtractor_Telecomftthbroadband (FTTH Fiber Gigabit Broadband Access).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.telecom_ftth_broadband.contract_renewal_barrier import ContractRenewalBarrierExtractor_Telecomftthbroadband
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_telecom_ftth_broadband_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Telecomftthbroadband()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_telecom_ftth_broadband_signal" in res.columns
    assert f"contract_renewal_barrier_telecom_ftth_broadband_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_telecom_ftth_broadband_signal"].isnull().any()

def test_contract_renewal_barrier_telecom_ftth_broadband_empty():
    extractor = ContractRenewalBarrierExtractor_Telecomftthbroadband()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
