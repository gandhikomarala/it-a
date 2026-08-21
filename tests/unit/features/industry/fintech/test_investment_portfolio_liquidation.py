# Unit Test for PortfolioLiquidation (fintech).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.fintech.investment_portfolio_liquidation import PortfolioLiquidation
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_investment_portfolio_liquidation_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PortfolioLiquidation()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"investment_portfolio_liquidation_signal" in res.columns
    assert f"investment_portfolio_liquidation_risk_index" in res.columns
    assert not res[f"investment_portfolio_liquidation_signal"].isnull().any()

def test_investment_portfolio_liquidation_empty_handling():
    extractor = PortfolioLiquidation()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
