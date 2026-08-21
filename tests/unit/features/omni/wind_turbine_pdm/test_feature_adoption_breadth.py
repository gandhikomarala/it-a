# Unit Test for FeatureAdoptionBreadthExtractor_Windturbinepdm (Offshore Wind Turbine Predictive Maintenance).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.wind_turbine_pdm.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Windturbinepdm
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_wind_turbine_pdm_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Windturbinepdm()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_wind_turbine_pdm_signal" in res.columns
    assert f"feature_adoption_breadth_wind_turbine_pdm_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_wind_turbine_pdm_signal"].isnull().any()

def test_feature_adoption_breadth_wind_turbine_pdm_empty():
    extractor = FeatureAdoptionBreadthExtractor_Windturbinepdm()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
