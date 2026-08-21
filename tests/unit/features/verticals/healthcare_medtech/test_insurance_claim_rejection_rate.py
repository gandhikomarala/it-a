# Unit Test for ClaimRejectionRateExtractor (Healthcare & MedTech SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.healthcare_medtech.insurance_claim_rejection_rate import ClaimRejectionRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_insurance_claim_rejection_rate_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ClaimRejectionRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"insurance_claim_rejection_rate_signal" in res.columns
    assert f"insurance_claim_rejection_rate_risk_score" in res.columns
    assert not res[f"insurance_claim_rejection_rate_signal"].isnull().any()

def test_insurance_claim_rejection_rate_empty_dataframe():
    extractor = ClaimRejectionRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
