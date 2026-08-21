# Comprehensive Unit Test for GreenEnergyTariffOptInExtractor (Energy & Smart Utilities).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.energy_utilities.green_energy_tariff_opt_in import GreenEnergyTariffOptInExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_green_energy_tariff_opt_in_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = GreenEnergyTariffOptInExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"green_energy_tariff_opt_in_signal" in res.columns
    assert f"green_energy_tariff_opt_in_risk_score" in res.columns
    assert not res[f"green_energy_tariff_opt_in_signal"].isnull().any()

def test_green_energy_tariff_opt_in_empty_handling():
    extractor = GreenEnergyTariffOptInExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
