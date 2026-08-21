# Unit Test for PredictiveWearVelocityExtractor_Nanotechmaterialslab (Advanced Nanomaterials Synthesis Lab).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.nanotech_materials_lab.predictive_wear_velocity import PredictiveWearVelocityExtractor_Nanotechmaterialslab
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_nanotech_materials_lab_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Nanotechmaterialslab()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_nanotech_materials_lab_signal" in res.columns
    assert f"predictive_wear_velocity_nanotech_materials_lab_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_nanotech_materials_lab_signal"].isnull().any()

def test_predictive_wear_velocity_nanotech_materials_lab_empty():
    extractor = PredictiveWearVelocityExtractor_Nanotechmaterialslab()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
