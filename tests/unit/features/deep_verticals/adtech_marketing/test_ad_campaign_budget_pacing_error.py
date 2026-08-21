# Comprehensive Unit Test for CampaignPacingErrorExtractor (AdTech & Growth Marketing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.adtech_marketing.ad_campaign_budget_pacing_error import CampaignPacingErrorExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ad_campaign_budget_pacing_error_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CampaignPacingErrorExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ad_campaign_budget_pacing_error_signal" in res.columns
    assert f"ad_campaign_budget_pacing_error_risk_score" in res.columns
    assert not res[f"ad_campaign_budget_pacing_error_signal"].isnull().any()

def test_ad_campaign_budget_pacing_error_empty():
    extractor = CampaignPacingErrorExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
