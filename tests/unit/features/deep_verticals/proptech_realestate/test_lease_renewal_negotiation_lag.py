# Comprehensive Unit Test for LeaseRenewalNegotiationLagExtractor (PropTech & Commercial Real Estate).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.proptech_realestate.lease_renewal_negotiation_lag import LeaseRenewalNegotiationLagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lease_renewal_negotiation_lag_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LeaseRenewalNegotiationLagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lease_renewal_negotiation_lag_signal" in res.columns
    assert f"lease_renewal_negotiation_lag_risk_score" in res.columns
    assert not res[f"lease_renewal_negotiation_lag_signal"].isnull().any()

def test_lease_renewal_negotiation_lag_empty():
    extractor = LeaseRenewalNegotiationLagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
