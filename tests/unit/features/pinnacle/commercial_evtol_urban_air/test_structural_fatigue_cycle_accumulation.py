# Unit Test for FatigueCycleAccumulationExtractor_Commercialevtolurbanair (All-Electric eVTOL Urban Air Mobility).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.commercial_evtol_urban_air.structural_fatigue_cycle_accumulation import FatigueCycleAccumulationExtractor_Commercialevtolurbanair
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_structural_fatigue_cycle_accumulation_commercial_evtol_urban_air_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FatigueCycleAccumulationExtractor_Commercialevtolurbanair()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"structural_fatigue_cycle_accumulation_commercial_evtol_urban_air_signal" in res.columns
    assert f"structural_fatigue_cycle_accumulation_commercial_evtol_urban_air_risk_score" in res.columns
    assert not res[f"structural_fatigue_cycle_accumulation_commercial_evtol_urban_air_signal"].isnull().any()

def test_structural_fatigue_cycle_accumulation_commercial_evtol_urban_air_empty():
    extractor = FatigueCycleAccumulationExtractor_Commercialevtolurbanair()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
