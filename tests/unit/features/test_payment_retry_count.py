# Comprehensive Unit Test for PaymentRetryCountExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.payment_retry_count import PaymentRetryCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_payment_retry_count_instantiation():
    extractor = PaymentRetryCountExtractor()
    assert extractor.prefix == "payment_retry_count"

def test_payment_retry_count_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = PaymentRetryCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("payment_retry_count_")]
    assert len(expected_cols) > 0

def test_payment_retry_count_transform_empty():
    extractor = PaymentRetryCountExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
