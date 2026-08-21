# Comprehensive Unit Test for ShiftTurnoverRateExtractor (Manufacturing & Industrial IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.manufacturing_iiot.production_shift_turnover_rate import ShiftTurnoverRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_production_shift_turnover_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ShiftTurnoverRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"production_shift_turnover_rate_signal" in res.columns
    assert f"production_shift_turnover_rate_risk_score" in res.columns
    assert not res[f"production_shift_turnover_rate_signal"].isnull().any()

def test_production_shift_turnover_rate_empty_handling():
    extractor = ShiftTurnoverRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
