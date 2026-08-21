# Unit Test for ContractRenewalBarrierExtractor_Commercialconstructionbim (Commercial BIM Construction Tracking).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.commercial_construction_bim.contract_renewal_barrier import ContractRenewalBarrierExtractor_Commercialconstructionbim
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_commercial_construction_bim_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Commercialconstructionbim()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_commercial_construction_bim_signal" in res.columns
    assert f"contract_renewal_barrier_commercial_construction_bim_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_commercial_construction_bim_signal"].isnull().any()

def test_contract_renewal_barrier_commercial_construction_bim_empty():
    extractor = ContractRenewalBarrierExtractor_Commercialconstructionbim()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
