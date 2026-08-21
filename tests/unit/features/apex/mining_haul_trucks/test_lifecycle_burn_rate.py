# Unit Test for LifecycleBurnRateExtractor_Mininghaultrucks (Autonomous Mining Haul Truck Fleets).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.mining_haul_trucks.lifecycle_burn_rate import LifecycleBurnRateExtractor_Mininghaultrucks
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_mining_haul_trucks_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Mininghaultrucks()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_mining_haul_trucks_signal" in res.columns
    assert f"lifecycle_burn_rate_mining_haul_trucks_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_mining_haul_trucks_signal"].isnull().any()

def test_lifecycle_burn_rate_mining_haul_trucks_empty():
    extractor = LifecycleBurnRateExtractor_Mininghaultrucks()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
