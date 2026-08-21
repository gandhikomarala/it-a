# Unit Test for CategoricalCardinalityValidator.
import pytest
import pandas as pd
from ml.data.quality_rules.categorical_cardinality_rule import CategoricalCardinalityValidator
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_categorical_cardinality_rule_validation():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    rule = CategoricalCardinalityValidator()
    result = rule.validate(df)
    
    assert isinstance(result, dict)
    assert "rule" in result
    assert "passed" in result
    assert result["records_evaluated"] == 50
