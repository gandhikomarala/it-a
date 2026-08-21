# Unit Test for ContractRenewalBarrierExtractor_Roboticsfleet (Autonomous Robotics & AMR Fleet Management).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.robotics_fleet.contract_renewal_barrier import ContractRenewalBarrierExtractor_Roboticsfleet
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_robotics_fleet_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Roboticsfleet()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_robotics_fleet_signal" in res.columns
    assert f"contract_renewal_barrier_robotics_fleet_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_robotics_fleet_signal"].isnull().any()

def test_contract_renewal_barrier_robotics_fleet_empty():
    extractor = ContractRenewalBarrierExtractor_Roboticsfleet()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
