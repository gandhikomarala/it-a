# Comprehensive Unit Test for SupportTicketSeverityIndexExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.support_ticket_severity_index import SupportTicketSeverityIndexExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_support_ticket_severity_index_instantiation():
    extractor = SupportTicketSeverityIndexExtractor()
    assert extractor.prefix == "support_ticket_severity_index"

def test_support_ticket_severity_index_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = SupportTicketSeverityIndexExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("support_ticket_severity_index_")]
    assert len(expected_cols) > 0

def test_support_ticket_severity_index_transform_empty():
    extractor = SupportTicketSeverityIndexExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
