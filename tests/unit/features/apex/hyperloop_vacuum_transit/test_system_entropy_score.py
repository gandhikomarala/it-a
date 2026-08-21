# Unit Test for SystemEntropyScoreExtractor_Hyperloopvacuumtransit (Hyperloop Low-Pressure Pneumatic Transit).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hyperloop_vacuum_transit.system_entropy_score import SystemEntropyScoreExtractor_Hyperloopvacuumtransit
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_hyperloop_vacuum_transit_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Hyperloopvacuumtransit()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_hyperloop_vacuum_transit_signal" in res.columns
    assert f"system_entropy_score_hyperloop_vacuum_transit_risk_score" in res.columns
    assert not res[f"system_entropy_score_hyperloop_vacuum_transit_signal"].isnull().any()

def test_system_entropy_score_hyperloop_vacuum_transit_empty():
    extractor = SystemEntropyScoreExtractor_Hyperloopvacuumtransit()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
