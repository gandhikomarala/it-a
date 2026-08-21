# Unit Test for FatigueCycleAccumulationExtractor_Neuromorphicretinavision (Event-Based Neuromorphic Silicon Retina).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.neuromorphic_retina_vision.structural_fatigue_cycle_accumulation import FatigueCycleAccumulationExtractor_Neuromorphicretinavision
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_structural_fatigue_cycle_accumulation_neuromorphic_retina_vision_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FatigueCycleAccumulationExtractor_Neuromorphicretinavision()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"structural_fatigue_cycle_accumulation_neuromorphic_retina_vision_signal" in res.columns
    assert f"structural_fatigue_cycle_accumulation_neuromorphic_retina_vision_risk_score" in res.columns
    assert not res[f"structural_fatigue_cycle_accumulation_neuromorphic_retina_vision_signal"].isnull().any()

def test_structural_fatigue_cycle_accumulation_neuromorphic_retina_vision_empty():
    extractor = FatigueCycleAccumulationExtractor_Neuromorphicretinavision()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
