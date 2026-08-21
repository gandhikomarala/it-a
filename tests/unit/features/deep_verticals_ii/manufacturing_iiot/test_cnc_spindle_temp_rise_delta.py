# Comprehensive Unit Test for CNCSpindleTempRiseExtractor (Manufacturing & Industrial IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.manufacturing_iiot.cnc_spindle_temp_rise_delta import CNCSpindleTempRiseExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_cnc_spindle_temp_rise_delta_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CNCSpindleTempRiseExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"cnc_spindle_temp_rise_delta_signal" in res.columns
    assert f"cnc_spindle_temp_rise_delta_risk_score" in res.columns
    assert not res[f"cnc_spindle_temp_rise_delta_signal"].isnull().any()

def test_cnc_spindle_temp_rise_delta_empty_handling():
    extractor = CNCSpindleTempRiseExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
