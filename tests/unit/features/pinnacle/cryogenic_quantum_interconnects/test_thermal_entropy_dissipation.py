# Unit Test for ThermalEntropyDissipationExtractor_Cryogenicquantuminterconnects (Cryogenic Millimeter-Wave Quantum Bus).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.cryogenic_quantum_interconnects.thermal_entropy_dissipation import ThermalEntropyDissipationExtractor_Cryogenicquantuminterconnects
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_thermal_entropy_dissipation_cryogenic_quantum_interconnects_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ThermalEntropyDissipationExtractor_Cryogenicquantuminterconnects()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"thermal_entropy_dissipation_cryogenic_quantum_interconnects_signal" in res.columns
    assert f"thermal_entropy_dissipation_cryogenic_quantum_interconnects_risk_score" in res.columns
    assert not res[f"thermal_entropy_dissipation_cryogenic_quantum_interconnects_signal"].isnull().any()

def test_thermal_entropy_dissipation_cryogenic_quantum_interconnects_empty():
    extractor = ThermalEntropyDissipationExtractor_Cryogenicquantuminterconnects()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
