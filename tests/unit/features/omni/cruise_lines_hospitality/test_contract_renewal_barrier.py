# Unit Test for ContractRenewalBarrierExtractor_Cruiselineshospitality (Cruise Line Passenger Lifetime Value).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cruise_lines_hospitality.contract_renewal_barrier import ContractRenewalBarrierExtractor_Cruiselineshospitality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_cruise_lines_hospitality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Cruiselineshospitality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_cruise_lines_hospitality_signal" in res.columns
    assert f"contract_renewal_barrier_cruise_lines_hospitality_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_cruise_lines_hospitality_signal"].isnull().any()

def test_contract_renewal_barrier_cruise_lines_hospitality_empty():
    extractor = ContractRenewalBarrierExtractor_Cruiselineshospitality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
