# Unit Test for SatisfactionDriftDeltaExtractor_Esportstournamentplatform (Esports Tournament & Streaming Platform).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.esports_tournament_platform.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Esportstournamentplatform
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_esports_tournament_platform_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Esportstournamentplatform()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_esports_tournament_platform_signal" in res.columns
    assert f"satisfaction_drift_delta_esports_tournament_platform_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_esports_tournament_platform_signal"].isnull().any()

def test_satisfaction_drift_delta_esports_tournament_platform_empty():
    extractor = SatisfactionDriftDeltaExtractor_Esportstournamentplatform()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
