# Unit Test for FreightRateBidWinRatioExtractor (Logistics & Supply Chain SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.logistics_freight.freight_rate_bid_win_ratio import FreightRateBidWinRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_freight_rate_bid_win_ratio_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FreightRateBidWinRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"freight_rate_bid_win_ratio_signal" in res.columns
    assert f"freight_rate_bid_win_ratio_risk_score" in res.columns
    assert not res[f"freight_rate_bid_win_ratio_signal"].isnull().any()

def test_freight_rate_bid_win_ratio_empty_dataframe():
    extractor = FreightRateBidWinRatioExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
