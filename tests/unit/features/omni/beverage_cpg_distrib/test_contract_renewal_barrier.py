# Unit Test for ContractRenewalBarrierExtractor_Beveragecpgdistrib (Beverage CPG Direct Store Delivery).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.beverage_cpg_distrib.contract_renewal_barrier import ContractRenewalBarrierExtractor_Beveragecpgdistrib
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_beverage_cpg_distrib_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Beveragecpgdistrib()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_beverage_cpg_distrib_signal" in res.columns
    assert f"contract_renewal_barrier_beverage_cpg_distrib_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_beverage_cpg_distrib_signal"].isnull().any()

def test_contract_renewal_barrier_beverage_cpg_distrib_empty():
    extractor = ContractRenewalBarrierExtractor_Beveragecpgdistrib()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
