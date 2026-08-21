# Comprehensive Unit Test for SubleaseInquiryExtractor (PropTech & Commercial Real Estate).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.proptech_realestate.sublease_inquiry_event_count import SubleaseInquiryExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_sublease_inquiry_event_count_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SubleaseInquiryExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"sublease_inquiry_event_count_signal" in res.columns
    assert f"sublease_inquiry_event_count_risk_score" in res.columns
    assert not res[f"sublease_inquiry_event_count_signal"].isnull().any()

def test_sublease_inquiry_event_count_empty():
    extractor = SubleaseInquiryExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
