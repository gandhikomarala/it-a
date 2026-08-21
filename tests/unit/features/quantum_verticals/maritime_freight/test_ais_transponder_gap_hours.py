# Comprehensive Unit Test for AISTransponderGapHoursExtractor (Maritime Shipping & Ocean Freight).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.maritime_freight.ais_transponder_gap_hours import AISTransponderGapHoursExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ais_transponder_gap_hours_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AISTransponderGapHoursExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ais_transponder_gap_hours_signal" in res.columns
    assert f"ais_transponder_gap_hours_risk_score" in res.columns
    assert not res[f"ais_transponder_gap_hours_signal"].isnull().any()

def test_ais_transponder_gap_hours_empty_handling():
    extractor = AISTransponderGapHoursExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
