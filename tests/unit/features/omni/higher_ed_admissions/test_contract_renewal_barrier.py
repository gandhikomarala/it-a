# Unit Test for ContractRenewalBarrierExtractor_Higheredadmissions (University Admissions & Enrollment).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.higher_ed_admissions.contract_renewal_barrier import ContractRenewalBarrierExtractor_Higheredadmissions
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_higher_ed_admissions_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Higheredadmissions()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_higher_ed_admissions_signal" in res.columns
    assert f"contract_renewal_barrier_higher_ed_admissions_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_higher_ed_admissions_signal"].isnull().any()

def test_contract_renewal_barrier_higher_ed_admissions_empty():
    extractor = ContractRenewalBarrierExtractor_Higheredadmissions()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
