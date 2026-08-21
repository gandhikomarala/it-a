# Comprehensive Unit Test for InfotainmentLTEDataBurnExtractor (Automotive & Connected Fleet).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.automotive_telematics.infotainment_lte_data_burn import InfotainmentLTEDataBurnExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_infotainment_lte_data_burn_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = InfotainmentLTEDataBurnExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"infotainment_lte_data_burn_signal" in res.columns
    assert f"infotainment_lte_data_burn_risk_score" in res.columns
    assert not res[f"infotainment_lte_data_burn_signal"].isnull().any()

def test_infotainment_lte_data_burn_empty_handling():
    extractor = InfotainmentLTEDataBurnExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
