# Unit Test for FleetTelematicsOfflineHoursExtractor (Logistics & Supply Chain SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.logistics_freight.fleet_telematics_offline_hours import FleetTelematicsOfflineHoursExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_fleet_telematics_offline_hours_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FleetTelematicsOfflineHoursExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"fleet_telematics_offline_hours_signal" in res.columns
    assert f"fleet_telematics_offline_hours_risk_score" in res.columns
    assert not res[f"fleet_telematics_offline_hours_signal"].isnull().any()

def test_fleet_telematics_offline_hours_empty_dataframe():
    extractor = FleetTelematicsOfflineHoursExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
