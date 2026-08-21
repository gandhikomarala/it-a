# Comprehensive Unit Test for SparePartsLeadDaysExtractor (Manufacturing & Industrial IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.manufacturing_iiot.spare_parts_replenishment_lead_days import SparePartsLeadDaysExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_spare_parts_replenishment_lead_days_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SparePartsLeadDaysExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"spare_parts_replenishment_lead_days_signal" in res.columns
    assert f"spare_parts_replenishment_lead_days_risk_score" in res.columns
    assert not res[f"spare_parts_replenishment_lead_days_signal"].isnull().any()

def test_spare_parts_replenishment_lead_days_empty_handling():
    extractor = SparePartsLeadDaysExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
