# Comprehensive Unit Test for GuildClanParticipationExtractor (Gaming & Interactive Entertainment).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.gaming_media.guild_clan_participation_index import GuildClanParticipationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_guild_clan_participation_index_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = GuildClanParticipationExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"guild_clan_participation_index_signal" in res.columns
    assert f"guild_clan_participation_index_risk_score" in res.columns
    assert not res[f"guild_clan_participation_index_signal"].isnull().any()

def test_guild_clan_participation_index_empty():
    extractor = GuildClanParticipationExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
