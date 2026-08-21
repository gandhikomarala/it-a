# Unit Test for ContractRenewalBarrierExtractor_Oilgaspipeline (Oil & Gas Pipeline Integrity).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.oil_gas_pipeline.contract_renewal_barrier import ContractRenewalBarrierExtractor_Oilgaspipeline
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_renewal_barrier_oil_gas_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ContractRenewalBarrierExtractor_Oilgaspipeline()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"contract_renewal_barrier_oil_gas_pipeline_signal" in res.columns
    assert f"contract_renewal_barrier_oil_gas_pipeline_risk_score" in res.columns
    assert not res[f"contract_renewal_barrier_oil_gas_pipeline_signal"].isnull().any()

def test_contract_renewal_barrier_oil_gas_pipeline_empty():
    extractor = ContractRenewalBarrierExtractor_Oilgaspipeline()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
