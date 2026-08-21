# Comprehensive Unit Test for PropertyManagerTurnoverExtractor (PropTech & Commercial Real Estate).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.proptech_realestate.property_manager_turnover_flag import PropertyManagerTurnoverExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_property_manager_turnover_flag_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PropertyManagerTurnoverExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"property_manager_turnover_flag_signal" in res.columns
    assert f"property_manager_turnover_flag_risk_score" in res.columns
    assert not res[f"property_manager_turnover_flag_signal"].isnull().any()

def test_property_manager_turnover_flag_empty():
    extractor = PropertyManagerTurnoverExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
