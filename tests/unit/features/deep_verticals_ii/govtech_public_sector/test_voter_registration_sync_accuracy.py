# Comprehensive Unit Test for VoterRegistrationSyncExtractor (GovTech & Municipal Services).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.govtech_public_sector.voter_registration_sync_accuracy import VoterRegistrationSyncExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_voter_registration_sync_accuracy_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VoterRegistrationSyncExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"voter_registration_sync_accuracy_signal" in res.columns
    assert f"voter_registration_sync_accuracy_risk_score" in res.columns
    assert not res[f"voter_registration_sync_accuracy_signal"].isnull().any()

def test_voter_registration_sync_accuracy_empty_handling():
    extractor = VoterRegistrationSyncExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
