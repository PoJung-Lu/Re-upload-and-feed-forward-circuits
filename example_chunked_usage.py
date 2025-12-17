"""
Example: Using Memory-Efficient Chunked Processing

This example demonstrates how to use jqc_nq_chunked() to train on large datasets
without running into memory issues.
"""

import numpy as np
from reupload_ff_circuit.q_circuits import qcircuit, test
from reupload_ff_circuit.util import initialize_params, initialize_data
from reupload_ff_circuit.q_functions import dm_generation

# Example 1: Direct usage of jqc_nq_chunked()
print("=" * 60)
print("Example 1: Direct Usage of jqc_nq_chunked()")
print("=" * 60)

# Initialize circuit
enc_dim, n_qubits, n_layers, n_reupload, n_rot = 3, 2, 2, 2, 2
num_class = 4

qc = qcircuit(enc_dim, n_qubits, n_layers, n_reupload, n_rot)
params = initialize_params(enc_dim, n_qubits, n_layers, n_reupload, n_rot, num_class)

# Generate sample data
X_train, y_train = initialize_data('squares', n_training=300, preprocess='scaling')
dm_labels = dm_generation('zyz', num_class)

print(f"\nDataset shape: {X_train.shape}")
print(f"Number of samples: {len(y_train)}")

# Standard method (memory-intensive for large datasets)
print("\n--- Using standard jqc_nq() ---")
result_standard = qc.jqc_nq(params, X_train.T, dm_labels[0])
print(f"Result shape: {result_standard.shape}")

# Chunked method (memory-efficient)
print("\n--- Using jqc_nq_chunked() with chunk_size=32 ---")
result_chunked = qc.jqc_nq_chunked(params, X_train.T, dm_labels[0], chunk_size=32)
print(f"Result shape: {result_chunked.shape}")

# Verify results are identical
print(f"\nResults match: {np.allclose(result_standard, result_chunked, rtol=1e-5)}")


# Example 2: Using chunked processing in training with test() function
print("\n" + "=" * 60)
print("Example 2: Using Chunked Processing in Training")
print("=" * 60)

# Setup for test() function
settings = (enc_dim, n_qubits, n_layers, n_reupload, n_rot)
shape = "bitwise"
Yc = np.array([[0, 1, 2, 3]] * len(y_train))

kwargs_base = {
    'qc': qc,
    'dm_labels': dm_labels,
    'num_class_1q': num_class,
    'shape': shape,
    'Yc': Yc,
    'noise': False
}

# Standard training
print("\n--- Standard training (without chunking) ---")
pred_standard, loss_standard, grad_standard = test(
    params, X_train, y_train, *settings, **kwargs_base
)
print(f"Loss: {loss_standard:.6f}")
print(f"Predictions shape: {pred_standard.shape}")

# Chunked training (recommended for large datasets)
print("\n--- Chunked training (memory-efficient) ---")
kwargs_chunked = {**kwargs_base, 'use_chunked': True, 'chunk_size': 32}
pred_chunked, loss_chunked, grad_chunked = test(
    params, X_train, y_train, *settings, **kwargs_chunked
)
print(f"Loss: {loss_chunked:.6f}")
print(f"Predictions shape: {pred_chunked.shape}")

# Verify losses match
print(f"\nLosses match: {np.allclose(loss_standard, loss_chunked, rtol=1e-5)}")


# Example 3: Memory monitoring with chunked processing
print("\n" + "=" * 60)
print("Example 3: Memory Monitoring")
print("=" * 60)

from reupload_ff_circuit.memory_monitor import MemoryMonitor

monitor = MemoryMonitor(threshold_mb=50.0)
monitor.start()

# Process with chunked method
result = qc.jqc_nq_chunked(params, X_train.T, dm_labels[0], chunk_size=16)
monitor.checkpoint("After chunked processing")

stats = monitor.stop()
print(f"\nMemory stats:")
print(f"  Peak memory: {stats['peak_mb']:.2f} MB")
print(f"  Memory delta: {stats['delta_mb']:+.2f} MB")


# Example 4: Recommended usage for large datasets
print("\n" + "=" * 60)
print("Example 4: Recommendations for Large Datasets")
print("=" * 60)

print("""
For datasets with memory issues, use these guidelines:

1. Small datasets (< 100 samples):
   - Use standard jqc_nq() - no chunking needed
   - Example: qc.jqc_nq(params, X.T, dm)

2. Medium datasets (100-500 samples):
   - Use chunked processing with chunk_size=32
   - Example: qc.jqc_nq_chunked(params, X.T, dm, chunk_size=32)

3. Large datasets (> 500 samples):
   - Use chunked processing with chunk_size=16 or smaller
   - Example: qc.jqc_nq_chunked(params, X.T, dm, chunk_size=16)

4. In training loop (using test() function):
   - Standard: test(params, X, y, *settings, **kwargs)
   - Chunked: test(params, X, y, *settings, use_chunked=True, chunk_size=32, **kwargs)

5. Memory monitoring:
   - Use memory_monitor.py to track memory usage
   - Adjust chunk_size based on available memory
""")

print("\nFor more details, see:")
print("  - reupload_ff_circuit/memory_monitor.py")
print("  - Demo_script_optimized.ipynb")
