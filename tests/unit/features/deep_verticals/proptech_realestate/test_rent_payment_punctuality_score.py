# Comprehensive Unit Test for RentPaymentPunctualityExtractor (PropTech & Commercial Real Estate).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.proptech_realestate.rent_payment_punctuality_score import RentPaymentPunctualityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_rent_payment_punctuality_score_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RentPaymentPunctualityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"rent_payment_punctuality_score_signal" in res.columns
    assert f"rent_payment_punctuality_score_risk_score" in res.columns
    assert not res[f"rent_payment_punctuality_score_signal"].isnull().any()

def test_rent_payment_punctuality_score_empty():
    extractor = RentPaymentPunctualityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
