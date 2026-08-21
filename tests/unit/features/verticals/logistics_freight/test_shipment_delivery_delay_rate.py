# Unit Test for ShipmentDeliveryDelayRateExtractor (Logistics & Supply Chain SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.logistics_freight.shipment_delivery_delay_rate import ShipmentDeliveryDelayRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_shipment_delivery_delay_rate_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ShipmentDeliveryDelayRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"shipment_delivery_delay_rate_signal" in res.columns
    assert f"shipment_delivery_delay_rate_risk_score" in res.columns
    assert not res[f"shipment_delivery_delay_rate_signal"].isnull().any()

def test_shipment_delivery_delay_rate_empty_dataframe():
    extractor = ShipmentDeliveryDelayRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
