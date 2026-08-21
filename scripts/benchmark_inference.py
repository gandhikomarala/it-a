#!/usr/bin/env python3
# Latency & throughput benchmarking script for ML inference.
import time
import numpy as np
import pandas as pd
from ml.data.synthetic_generator import SyntheticCustomerGenerator
from ml.training.orchestrator import TrainingOrchestrator

def benchmark():
    print("Generating benchmark dataset...")
    gen = SyntheticCustomerGenerator()
    df = gen.generate(1000)

    print("Training model for benchmark...")
    pipeline, model, metrics = TrainingOrchestrator.train_and_evaluate(df, algorithm="LightGBM")

    print("Benchmarking single prediction latency over 1,000 iterations...")
    latencies_ms = []
    test_instance = df.drop(columns=["churn"]).iloc[0:1]

    for _ in range(1000):
        t0 = time.perf_counter()
        X_trans = pipeline.transform(test_instance)
        prob = model.predict_proba(X_trans)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat)

    arr = np.array(latencies_ms)
    print("=" * 60)
    print("  SINGLE INFERENCE LATENCY BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Mean Latency : {np.mean(arr):.2f} ms")
    print(f"P50 Latency  : {np.percentile(arr, 50):.2f} ms")
    print(f"P90 Latency  : {np.percentile(arr, 90):.2f} ms")
    print(f"P99 Latency  : {np.percentile(arr, 99):.2f} ms")
    print(f"Throughput   : {1000.0 / np.mean(arr):.1f} predictions / sec (single-core)")
    print("=" * 60)

if __name__ == '__main__':
    benchmark()
