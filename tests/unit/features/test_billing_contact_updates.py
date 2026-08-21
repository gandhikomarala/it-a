# Comprehensive Unit Test for BillingContactUpdateExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.billing_contact_updates import BillingContactUpdateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_billing_contact_updates_instantiation():
    extractor = BillingContactUpdateExtractor()
    assert extractor.prefix == "billing_contact_updates"

def test_billing_contact_updates_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = BillingContactUpdateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("billing_contact_updates_")]
    assert len(expected_cols) > 0

def test_billing_contact_updates_transform_empty():
    extractor = BillingContactUpdateExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
