# Comprehensive Unit Test for FOIARequestFulfillmentLagExtractor (GovTech & Municipal Services).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.govtech_public_sector.foia_request_fulfillment_lag import FOIARequestFulfillmentLagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_foia_request_fulfillment_lag_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FOIARequestFulfillmentLagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"foia_request_fulfillment_lag_signal" in res.columns
    assert f"foia_request_fulfillment_lag_risk_score" in res.columns
    assert not res[f"foia_request_fulfillment_lag_signal"].isnull().any()

def test_foia_request_fulfillment_lag_empty_handling():
    extractor = FOIARequestFulfillmentLagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
