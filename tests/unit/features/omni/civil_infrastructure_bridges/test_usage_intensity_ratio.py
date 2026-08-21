# Unit Test for UsageIntensityRatioExtractor_Civilinfrastructurebridges (Civil Infrastructure & Bridge Health).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.civil_infrastructure_bridges.usage_intensity_ratio import UsageIntensityRatioExtractor_Civilinfrastructurebridges
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_intensity_ratio_civil_infrastructure_bridges_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UsageIntensityRatioExtractor_Civilinfrastructurebridges()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"usage_intensity_ratio_civil_infrastructure_bridges_signal" in res.columns
    assert f"usage_intensity_ratio_civil_infrastructure_bridges_risk_score" in res.columns
    assert not res[f"usage_intensity_ratio_civil_infrastructure_bridges_signal"].isnull().any()

def test_usage_intensity_ratio_civil_infrastructure_bridges_empty():
    extractor = UsageIntensityRatioExtractor_Civilinfrastructurebridges()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
