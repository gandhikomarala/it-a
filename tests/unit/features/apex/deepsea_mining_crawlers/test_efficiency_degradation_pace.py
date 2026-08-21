# Unit Test for EfficiencyDegradationPaceExtractor_Deepseaminingcrawlers (Abyssal Plain Polymetallic Nodule Harvesters).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.deepsea_mining_crawlers.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Deepseaminingcrawlers
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_deepsea_mining_crawlers_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Deepseaminingcrawlers()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_deepsea_mining_crawlers_signal" in res.columns
    assert f"efficiency_degradation_pace_deepsea_mining_crawlers_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_deepsea_mining_crawlers_signal"].isnull().any()

def test_efficiency_degradation_pace_deepsea_mining_crawlers_empty():
    extractor = EfficiencyDegradationPaceExtractor_Deepseaminingcrawlers()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
