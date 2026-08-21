# Comprehensive Unit Test for ListingInquiryDecayExtractor (PropTech & Commercial Real Estate).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.proptech_realestate.property_listing_inquiry_decay import ListingInquiryDecayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_property_listing_inquiry_decay_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ListingInquiryDecayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"property_listing_inquiry_decay_signal" in res.columns
    assert f"property_listing_inquiry_decay_risk_score" in res.columns
    assert not res[f"property_listing_inquiry_decay_signal"].isnull().any()

def test_property_listing_inquiry_decay_empty():
    extractor = ListingInquiryDecayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
