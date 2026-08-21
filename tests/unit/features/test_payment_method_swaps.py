# Comprehensive Unit Test for PaymentMethodSwapExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.payment_method_swaps import PaymentMethodSwapExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_payment_method_swaps_instantiation():
    extractor = PaymentMethodSwapExtractor()
    assert extractor.prefix == "payment_method_swaps"

def test_payment_method_swaps_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = PaymentMethodSwapExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("payment_method_swaps_")]
    assert len(expected_cols) > 0

def test_payment_method_swaps_transform_empty():
    extractor = PaymentMethodSwapExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
