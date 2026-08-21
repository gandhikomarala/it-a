# Comprehensive Unit Test for HarvestLogisticsTruckDelayExtractor (Agriculture & Precision Farming).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.agriculture_agtech.harvest_logistics_truck_delay import HarvestLogisticsTruckDelayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harvest_logistics_truck_delay_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarvestLogisticsTruckDelayExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harvest_logistics_truck_delay_signal" in res.columns
    assert f"harvest_logistics_truck_delay_risk_score" in res.columns
    assert not res[f"harvest_logistics_truck_delay_signal"].isnull().any()

def test_harvest_logistics_truck_delay_empty_handling():
    extractor = HarvestLogisticsTruckDelayExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
