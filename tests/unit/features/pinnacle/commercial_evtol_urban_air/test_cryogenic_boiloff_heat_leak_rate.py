# Unit Test for HeatLeakRateExtractor_Commercialevtolurbanair (All-Electric eVTOL Urban Air Mobility).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.commercial_evtol_urban_air.cryogenic_boiloff_heat_leak_rate import HeatLeakRateExtractor_Commercialevtolurbanair
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_cryogenic_boiloff_heat_leak_rate_commercial_evtol_urban_air_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HeatLeakRateExtractor_Commercialevtolurbanair()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"cryogenic_boiloff_heat_leak_rate_commercial_evtol_urban_air_signal" in res.columns
    assert f"cryogenic_boiloff_heat_leak_rate_commercial_evtol_urban_air_risk_score" in res.columns
    assert not res[f"cryogenic_boiloff_heat_leak_rate_commercial_evtol_urban_air_signal"].isnull().any()

def test_cryogenic_boiloff_heat_leak_rate_commercial_evtol_urban_air_empty():
    extractor = HeatLeakRateExtractor_Commercialevtolurbanair()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
