# Comprehensive Unit Test for HotelUpgradeVelocityExtractor (Travel, Airline & Hospitality).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.travel_hospitality.hotel_room_upgrade_velocity import HotelUpgradeVelocityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_hotel_room_upgrade_velocity_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HotelUpgradeVelocityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"hotel_room_upgrade_velocity_signal" in res.columns
    assert f"hotel_room_upgrade_velocity_risk_score" in res.columns
    assert not res[f"hotel_room_upgrade_velocity_signal"].isnull().any()

def test_hotel_room_upgrade_velocity_empty_handling():
    extractor = HotelUpgradeVelocityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
