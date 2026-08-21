# Unit Test for ContractRenewalBarrierExtractor_Musicstreamingroyalties (Music Streaming Artist Royalty Split).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.music_streaming_royalties.contract_renewal_barrier import ContractRenewalBarrierExtractor_Musicstreamingroyalties
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_music_streaming_royalties_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Musicstreamingroyalties()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_music_streaming_royalties_signal" in res.columns
    assert f"contract_renewal_barrier_music_streaming_royalties_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_music_streaming_royalties_signal"].isnull().any()

def test_contract_renewal_barrier_music_streaming_royalties_empty():
    extractor = ContractRenewalBarrierExtractor_Musicstreamingroyalties()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
