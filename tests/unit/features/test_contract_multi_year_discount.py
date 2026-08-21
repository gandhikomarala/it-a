# Comprehensive Unit Test for ContractMultiYearDiscountExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.contract_multi_year_discount import ContractMultiYearDiscountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_multi_year_discount_instantiation():
    extractor = ContractMultiYearDiscountExtractor()
    assert extractor.prefix == "contract_multi_year_discount"

def test_contract_multi_year_discount_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = ContractMultiYearDiscountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("contract_multi_year_discount_")]
    assert len(expected_cols) > 0

def test_contract_multi_year_discount_transform_empty():
    extractor = ContractMultiYearDiscountExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
