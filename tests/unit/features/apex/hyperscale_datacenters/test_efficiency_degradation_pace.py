# Unit Test for EfficiencyDegradationPaceExtractor_Hyperscaledatacenters (Hyperscale Datacenter Liquid Cooling).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hyperscale_datacenters.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Hyperscaledatacenters
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_hyperscale_datacenters_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Hyperscaledatacenters()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_hyperscale_datacenters_signal" in res.columns
    assert f"efficiency_degradation_pace_hyperscale_datacenters_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_hyperscale_datacenters_signal"].isnull().any()

def test_efficiency_degradation_pace_hyperscale_datacenters_empty():
    extractor = EfficiencyDegradationPaceExtractor_Hyperscaledatacenters()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
