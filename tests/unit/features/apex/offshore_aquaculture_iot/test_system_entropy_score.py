# Unit Test for SystemEntropyScoreExtractor_Offshoreaquacultureiot (Open-Ocean Smart Aquaculture Cages).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.offshore_aquaculture_iot.system_entropy_score import SystemEntropyScoreExtractor_Offshoreaquacultureiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_offshore_aquaculture_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Offshoreaquacultureiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_offshore_aquaculture_iot_signal" in res.columns
    assert f"system_entropy_score_offshore_aquaculture_iot_risk_score" in res.columns
    assert not res[f"system_entropy_score_offshore_aquaculture_iot_signal"].isnull().any()

def test_system_entropy_score_offshore_aquaculture_iot_empty():
    extractor = SystemEntropyScoreExtractor_Offshoreaquacultureiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
