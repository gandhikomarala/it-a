# Unit Test for TelemetryStabilityIndexExtractor_Cryogenicliquidhydrogen (Cryogenic Liquid Hydrogen Transport & Storage).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.cryogenic_liquid_hydrogen.telemetry_stability_index import TelemetryStabilityIndexExtractor_Cryogenicliquidhydrogen
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_cryogenic_liquid_hydrogen_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Cryogenicliquidhydrogen()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_cryogenic_liquid_hydrogen_signal" in res.columns
    assert f"telemetry_stability_index_cryogenic_liquid_hydrogen_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_cryogenic_liquid_hydrogen_signal"].isnull().any()

def test_telemetry_stability_index_cryogenic_liquid_hydrogen_empty():
    extractor = TelemetryStabilityIndexExtractor_Cryogenicliquidhydrogen()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
