# Comprehensive Unit Test for ContractAutoRenewStatusExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.contract_auto_renew_status import ContractAutoRenewStatusExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_auto_renew_status_instantiation():
    extractor = ContractAutoRenewStatusExtractor()
    assert extractor.prefix == "contract_auto_renew_status"

def test_contract_auto_renew_status_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = ContractAutoRenewStatusExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("contract_auto_renew_status_")]
    assert len(expected_cols) > 0

def test_contract_auto_renew_status_transform_empty():
    extractor = ContractAutoRenewStatusExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
