# Unit Test for PredictiveWearVelocityExtractor_Cryogenicliquidhydrogen (Cryogenic Liquid Hydrogen Transport & Storage).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.cryogenic_liquid_hydrogen.predictive_wear_velocity import PredictiveWearVelocityExtractor_Cryogenicliquidhydrogen
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_cryogenic_liquid_hydrogen_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Cryogenicliquidhydrogen()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_cryogenic_liquid_hydrogen_signal" in res.columns
    assert f"predictive_wear_velocity_cryogenic_liquid_hydrogen_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_cryogenic_liquid_hydrogen_signal"].isnull().any()

def test_predictive_wear_velocity_cryogenic_liquid_hydrogen_empty():
    extractor = PredictiveWearVelocityExtractor_Cryogenicliquidhydrogen()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
