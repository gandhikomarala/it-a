# Unit Test for ContractRenewalBarrierExtractor_Waterutilityami (Municipal Smart Water AMI Network).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.water_utility_ami.contract_renewal_barrier import ContractRenewalBarrierExtractor_Waterutilityami
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_water_utility_ami_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Waterutilityami()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_water_utility_ami_signal" in res.columns
    assert f"contract_renewal_barrier_water_utility_ami_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_water_utility_ami_signal"].isnull().any()

def test_contract_renewal_barrier_water_utility_ami_empty():
    extractor = ContractRenewalBarrierExtractor_Waterutilityami()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
