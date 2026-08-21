# Comprehensive Unit Test for ContractDaysToExpirationExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.contract_days_to_expiration import ContractDaysToExpirationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_days_to_expiration_instantiation():
    extractor = ContractDaysToExpirationExtractor()
    assert extractor.prefix == "contract_days_to_expiration"

def test_contract_days_to_expiration_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = ContractDaysToExpirationExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("contract_days_to_expiration_")]
    assert len(expected_cols) > 0

def test_contract_days_to_expiration_transform_empty():
    extractor = ContractDaysToExpirationExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
