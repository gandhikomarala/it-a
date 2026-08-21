# Unit Test for ContractRenewalBarrierExtractor_Windturbinepdm (Offshore Wind Turbine Predictive Maintenance).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.wind_turbine_pdm.contract_renewal_barrier import ContractRenewalBarrierExtractor_Windturbinepdm
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_wind_turbine_pdm_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Windturbinepdm()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_wind_turbine_pdm_signal" in res.columns
    assert f"contract_renewal_barrier_wind_turbine_pdm_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_wind_turbine_pdm_signal"].isnull().any()

def test_contract_renewal_barrier_wind_turbine_pdm_empty():
    extractor = ContractRenewalBarrierExtractor_Windturbinepdm()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
