# Unit Test for SystemEntropyScoreExtractor_Carbondirectaircapture (Direct Air Carbon Capture & Sequestration).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.carbon_direct_air_capture.system_entropy_score import SystemEntropyScoreExtractor_Carbondirectaircapture
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_carbon_direct_air_capture_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Carbondirectaircapture()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_carbon_direct_air_capture_signal" in res.columns
    assert f"system_entropy_score_carbon_direct_air_capture_risk_score" in res.columns
    assert not res[f"system_entropy_score_carbon_direct_air_capture_signal"].isnull().any()

def test_system_entropy_score_carbon_direct_air_capture_empty():
    extractor = SystemEntropyScoreExtractor_Carbondirectaircapture()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
