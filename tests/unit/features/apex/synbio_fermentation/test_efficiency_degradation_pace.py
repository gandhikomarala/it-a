# Unit Test for EfficiencyDegradationPaceExtractor_Synbiofermentation (Synthetic Biology & Bioreactor Fermentation).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.synbio_fermentation.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Synbiofermentation
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_synbio_fermentation_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Synbiofermentation()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_synbio_fermentation_signal" in res.columns
    assert f"efficiency_degradation_pace_synbio_fermentation_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_synbio_fermentation_signal"].isnull().any()

def test_efficiency_degradation_pace_synbio_fermentation_empty():
    extractor = EfficiencyDegradationPaceExtractor_Synbiofermentation()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
