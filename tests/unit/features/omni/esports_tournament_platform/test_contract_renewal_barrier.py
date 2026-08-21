# Unit Test for ContractRenewalBarrierExtractor_Esportstournamentplatform (Esports Tournament & Streaming Platform).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.esports_tournament_platform.contract_renewal_barrier import ContractRenewalBarrierExtractor_Esportstournamentplatform
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_esports_tournament_platform_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Esportstournamentplatform()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_esports_tournament_platform_signal" in res.columns
    assert f"contract_renewal_barrier_esports_tournament_platform_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_esports_tournament_platform_signal"].isnull().any()

def test_contract_renewal_barrier_esports_tournament_platform_empty():
    extractor = ContractRenewalBarrierExtractor_Esportstournamentplatform()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
