# Unit Test for LifecycleBurnRateExtractor_Droneswarmmeshnetworks (Autonomous Drone Swarm Ad-Hoc Mesh).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.drone_swarm_mesh_networks.lifecycle_burn_rate import LifecycleBurnRateExtractor_Droneswarmmeshnetworks
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_drone_swarm_mesh_networks_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Droneswarmmeshnetworks()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_drone_swarm_mesh_networks_signal" in res.columns
    assert f"lifecycle_burn_rate_drone_swarm_mesh_networks_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_drone_swarm_mesh_networks_signal"].isnull().any()

def test_lifecycle_burn_rate_drone_swarm_mesh_networks_empty():
    extractor = LifecycleBurnRateExtractor_Droneswarmmeshnetworks()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
