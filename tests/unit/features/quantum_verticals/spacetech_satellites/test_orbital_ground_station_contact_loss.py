# Comprehensive Unit Test for GroundStationContactLossExtractor (SpaceTech & LEO Satellite Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.spacetech_satellites.orbital_ground_station_contact_loss import GroundStationContactLossExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_orbital_ground_station_contact_loss_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = GroundStationContactLossExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"orbital_ground_station_contact_loss_signal" in res.columns
    assert f"orbital_ground_station_contact_loss_risk_score" in res.columns
    assert not res[f"orbital_ground_station_contact_loss_signal"].isnull().any()

def test_orbital_ground_station_contact_loss_empty_handling():
    extractor = GroundStationContactLossExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
