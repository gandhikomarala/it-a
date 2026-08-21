# Unit Test for SystemEntropyScoreExtractor_Wastetoenergygasification (Plasma Gasification Waste-to-Energy).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.waste_to_energy_gasification.system_entropy_score import SystemEntropyScoreExtractor_Wastetoenergygasification
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_waste_to_energy_gasification_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Wastetoenergygasification()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_waste_to_energy_gasification_signal" in res.columns
    assert f"system_entropy_score_waste_to_energy_gasification_risk_score" in res.columns
    assert not res[f"system_entropy_score_waste_to_energy_gasification_signal"].isnull().any()

def test_system_entropy_score_waste_to_energy_gasification_empty():
    extractor = SystemEntropyScoreExtractor_Wastetoenergygasification()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
