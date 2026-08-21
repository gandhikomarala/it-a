# Unit Test for PlasmaInstabilityGrowthExtractor_Deepspaceopticalcomms (Deep Space Optical Laser Communications).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.deep_space_optical_comms.plasma_instability_growth_rate import PlasmaInstabilityGrowthExtractor_Deepspaceopticalcomms
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_plasma_instability_growth_rate_deep_space_optical_comms_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PlasmaInstabilityGrowthExtractor_Deepspaceopticalcomms()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"plasma_instability_growth_rate_deep_space_optical_comms_signal" in res.columns
    assert f"plasma_instability_growth_rate_deep_space_optical_comms_risk_score" in res.columns
    assert not res[f"plasma_instability_growth_rate_deep_space_optical_comms_signal"].isnull().any()

def test_plasma_instability_growth_rate_deep_space_optical_comms_empty():
    extractor = PlasmaInstabilityGrowthExtractor_Deepspaceopticalcomms()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
