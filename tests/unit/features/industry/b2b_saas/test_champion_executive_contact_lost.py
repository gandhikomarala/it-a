# Unit Test for ChampionContactLost (b2b_saas).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.b2b_saas.champion_executive_contact_lost import ChampionContactLost
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_champion_executive_contact_lost_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ChampionContactLost()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"champion_executive_contact_lost_signal" in res.columns
    assert f"champion_executive_contact_lost_risk_index" in res.columns
    assert not res[f"champion_executive_contact_lost_signal"].isnull().any()

def test_champion_executive_contact_lost_empty_handling():
    extractor = ChampionContactLost()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
