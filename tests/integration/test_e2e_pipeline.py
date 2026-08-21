# End-to-end integration test for full training & inference lifecycle.
import pytest
from ml.data.synthetic_generator import SyntheticCustomerGenerator
from ml.pipelines.training_pipeline import FullTrainingPipeline
from ml.inference.single_engine import SingleInferenceEngine
from packages.schemas.prediction import SinglePredictionRequest

def test_full_mlops_lifecycle_e2e():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(300)

    # 1. Run experiment
    res = FullTrainingPipeline.run_experiment(
        df=df,
        experiment_name="Integration_Test_Exp",
        algorithms=["LogisticRegression", "LightGBM"],
        training_mode="FAST"
    )
    assert res["best_algorithm"] in ["LogisticRegression", "LightGBM"]
    assert res["best_metrics"].roc_auc > 0.60

    # 2. Run inference
    best_pipe = res["best_metrics"]
    assert res["artifact_info"]["checksum_sha256"] is not None
