# autograd-cpp

![bs with fur](imgs/default-mammal.jpeg)

A custom C++17 / CUDA deep learning framework with Pybind11 Python bindings, built from scratch with a reverse-mode autograd engine, GPU-native neural network layers, optimizers, and data loaders.

## Architecture Overview

- **Autograd & Tensor Engine (`src/autograd/Tensor.h`, `Tensor.cu`)**
  - Dynamic reverse-mode computational graphs on **CPU (OpenMP)** and **CUDA (cuBLAS + custom kernels)**
  - N-dimensional row-major strided storage, NumPy-style broadcasting, `contiguous()`, `view()`, `reshape()`, `permute()`, `squeeze()`, `unsqueeze()`, and `expand()`
  - Elementwise ops (`+`, `-`, `*`, `/`, `pow`, `sqrt`, `exp`, `log`, `tanh`, `sigmoid`, `relu`), reductions (`sum`, `mean`, `max`, `min`, `argmax`, `argmin`, `argsort`), and `softmax` / `log_softmax`
- **Neural Network Layers (`src/nn/`)**
  - `Module` base class with recursive `parameters()`, `train()` / `eval()`, `zero_grad()`, and `to(Device)` (migrating both parameters and non-learnable tracking buffers)
  - `Sequential`, `Linear`, `MLP`
  - `Conv2D` (`im2col` / `col2im` + cuBLAS GEMM), `MaxPool2D`, `AvgPool2D`, `BatchNorm2D` (EMA running stats + eval mode)
  - `Dropout` and spatial channel `Dropout2D` (stateless SplitMix64 GPU/CPU masks, `eval()` & `p=0` zero-copy pass-through)
- **Losses & Optimizers (`src/nn/Loss.h`, `Loss.cu`, `src/optim/`)**
  - `MSELoss`, `CrossEntropyLoss`, and `SparseCategoricalCrossEntropyLoss` (fused shared-memory CUDA reduction with zero per-step `cudaMalloc` / host sync)
  - GPU-native `SGD` (momentum, weight decay) and `Adam` (weight decay, AMSGrad, maximize)
- **Data Pipeline (`src/data/Dataset.h`, `DataLoader.h`)**
  - Abstract `Dataset`, `TensorDataset`, and iterable `DataLoader` (C++20 `std::input_iterator`-compliant, isolated per-iterator `std::mt19937` epoch shuffling, `drop_last`, single H2D mini-batch collation)
- **Python Bindings (`src/autograd/bindings.cpp`)**
  - `autograd_cpp` Pybind11 module with contiguous buffer size checks, `PyModule` / `PySequential` / `PyDataset` trampolines, and `py::keep_alive` GC lifetime safety

## Build & Test

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the Python verification scripts (with `build/` built):

```bash
python3 tests/pytorch-tests/autograd_verification.py
python3 tests/pytorch-tests/linear_verification.py
python3 tests/pytorch-tests/mlp_verification.py
python3 tests/pytorch-tests/cnn_gpu_verification.py
python3 tests/pytorch-tests/phase5_3_5_interop_verification.py
```

## Verification Summary (RTX 3050 / WSL2)

- **C++ / CUDA CTest Suite (`ctest --output-on-failure`):** **4/4 suites passed**
  - `TensorGradMatrixTest`: **27** `[PASS]` checks (CPU autograd, views, broadcasting, reductions)
  - `TensorGradGPUTest`: **48** GPU test suites / **52** `[PASS]` checks (CUDA kernels, Conv2D, Pool2D, BatchNorm2D, SparseCE, SGD/Adam, Dropout, Dropout2D, Dataset/DataLoader leak & concurrency tests)
  - `NNIntegrationTest`: **11** `[PASS]` checks
  - `GPUNNIntegrationTest`: **1** `[PASS]` end-to-end GPU ConvNet convergence check
- **Python / C++ Interop Suite:** All 5 verification scripts passed (`NaN` guards, `Sequential` submodule propagation, `PyModule` virtual overrides, `gc.collect()` lifetime safety, and concurrent `DataLoader` iterators).

## Fashion-MNIST GPU Benchmark

5-epoch training of `FashionCNN` (`Conv2D` + `BatchNorm2D` + `MaxPool2D` + `Dropout2D(0.1)` + `Linear` + `Dropout(0.25)`, **207,018** trainable parameters) using `TensorDataset`, CUDA `DataLoader` (`batch_size=256`, `shuffle=True`), `SparseCategoricalCrossEntropyLoss`, and GPU `Adam` (`lr=0.002`, `weight_decay=1e-4`) on **5,000** training and **1,000** test images:

| Metric | Result (RTX 3050) |
| --- | --- |
| Epoch 5 Train Loss / Accuracy | `0.4509` / `83.30%` |
| Training Throughput (5 epochs) | `603.7 img/s` (`41.41 s` total) |
| Test Accuracy (1,000 unseen images) | **`84.80%`** |
| Test Inference Latency | `734.50 ms` (`734.5 µs/image`) |
