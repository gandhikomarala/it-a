# Comprehensive Unit Test for InvoiceDownloadFrequencyExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.invoice_download_frequency import InvoiceDownloadFrequencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_invoice_download_frequency_instantiation():
    extractor = InvoiceDownloadFrequencyExtractor()
    assert extractor.prefix == "invoice_download_frequency"

def test_invoice_download_frequency_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = InvoiceDownloadFrequencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("invoice_download_frequency_")]
    assert len(expected_cols) > 0

def test_invoice_download_frequency_transform_empty():
    extractor = InvoiceDownloadFrequencyExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
