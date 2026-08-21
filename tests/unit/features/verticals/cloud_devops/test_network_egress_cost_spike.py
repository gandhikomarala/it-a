# Unit Test for NetworkEgressCostSpikeExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.network_egress_cost_spike import NetworkEgressCostSpikeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_network_egress_cost_spike_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = NetworkEgressCostSpikeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"network_egress_cost_spike_signal" in res.columns
    assert f"network_egress_cost_spike_risk_score" in res.columns
    assert not res[f"network_egress_cost_spike_signal"].isnull().any()

def test_network_egress_cost_spike_empty_dataframe():
    extractor = NetworkEgressCostSpikeExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
