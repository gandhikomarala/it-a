# Unit Test for ContractRenewalBarrierExtractor_Mvnocellulartelecom (MVNO Mobile Virtual Network Operator).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.mvno_cellular_telecom.contract_renewal_barrier import ContractRenewalBarrierExtractor_Mvnocellulartelecom
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_mvno_cellular_telecom_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Mvnocellulartelecom()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_mvno_cellular_telecom_signal" in res.columns
    assert f"contract_renewal_barrier_mvno_cellular_telecom_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_mvno_cellular_telecom_signal"].isnull().any()

def test_contract_renewal_barrier_mvno_cellular_telecom_empty():
    extractor = ContractRenewalBarrierExtractor_Mvnocellulartelecom()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
