# Unit Test for FeatureAdoptionBreadthExtractor_Hydroelectricdamiot (Hydroelectric Dam Structural Health).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.hydroelectric_dam_iot.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Hydroelectricdamiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_hydroelectric_dam_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Hydroelectricdamiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_hydroelectric_dam_iot_signal" in res.columns
    assert f"feature_adoption_breadth_hydroelectric_dam_iot_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_hydroelectric_dam_iot_signal"].isnull().any()

def test_feature_adoption_breadth_hydroelectric_dam_iot_empty():
    extractor = FeatureAdoptionBreadthExtractor_Hydroelectricdamiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
