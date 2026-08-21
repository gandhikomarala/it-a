# Comprehensive Unit Test for PortCongestionDwellHoursExtractor (Maritime Shipping & Ocean Freight).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.maritime_freight.vessel_port_congestion_dwell_hours import PortCongestionDwellHoursExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_vessel_port_congestion_dwell_hours_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PortCongestionDwellHoursExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"vessel_port_congestion_dwell_hours_signal" in res.columns
    assert f"vessel_port_congestion_dwell_hours_risk_score" in res.columns
    assert not res[f"vessel_port_congestion_dwell_hours_signal"].isnull().any()

def test_vessel_port_congestion_dwell_hours_empty_handling():
    extractor = PortCongestionDwellHoursExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
