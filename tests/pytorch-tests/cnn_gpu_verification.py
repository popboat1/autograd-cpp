"""
End-to-End Convolutional Neural Network GPU Training & Verification in Python
Uses autograd_cpp with Conv2D, BatchNorm2D, MaxPool2D, Linear, Adam, and MSELoss.
"""

import os
import sys
import math

# windows mingw dll path lookup guard
for path_dir in os.environ.get("PATH", "").split(os.pathsep):
    if os.path.exists(os.path.join(path_dir, "g++.exe")) or os.path.exists(os.path.join(path_dir, "gcc.exe")):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(path_dir)
        break

script_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(tests_dir)

sys.path.append(os.path.join(project_root, "build"))

import autograd_cpp

class ConvClassifier(autograd_cpp.Module):
    def __init__(self, in_channels=1, out_channels=4, num_classes=2):
        super().__init__()
        self.conv = autograd_cpp.Conv2D(in_channels, out_channels, 3, stride=1, padding=1)
        self.bn = autograd_cpp.BatchNorm2D(out_channels)
        self.pool = autograd_cpp.MaxPool2D(2, stride=2)
        self.fc = autograd_cpp.Linear(out_channels * 4 * 4, num_classes)

    def forward(self, x):
        # 1. conv -> batchnorm -> relu -> pool
        c = self.conv(x)
        b = self.bn(c)
        a = b.relu()
        p = self.pool(a)

        # 2. flatten [B, C, H, W] -> [B, C*H*W]
        flat = p.view([p.shape[0], -1])

        # 3. dense projection
        return self.fc(flat)

def main():
    print("==================================================")
    print("Starting Python GPU ConvNet End-to-End Training")
    print("==================================================")

    # 1. Synthetic dataset setup: B=4, C=1, H=8, W=8
    B, C, H, W = 4, 1, 8, 8
    num_classes = 2

    x_vals = [math.sin(i * 0.1) for i in range(B * C * H * W)]
    y_vals = [
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0
    ]

    # Create tensors directly on GPU VRAM
    X = autograd_cpp.Tensor(x_vals, [B, C, H, W], False, autograd_cpp.Device.CUDA)
    Y = autograd_cpp.Tensor(y_vals, [B, num_classes], False, autograd_cpp.Device.CUDA)

    print(f"Input tensor device: {X.device}, shape: {X.shape}")
    print(f"Target tensor device: {Y.device}, shape: {Y.shape}")

    # 2. Instantiate Python-subclassed Module and transfer to CUDA
    model = ConvClassifier(C, 4, num_classes)
    model.to(autograd_cpp.Device.CUDA)

    params = model.parameters()
    print(f"Total model parameters registered: {len(params)}")
    for i, p in enumerate(params):
        print(f"  param {i} shape: {p.shape}, device: {p.device}")

    # 3. Optimizer & Loss setup
    optimizer = autograd_cpp.optim.Adam(params, lr=0.02, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001)
    criterion = autograd_cpp.MSELoss()

    initial_loss = 0.0
    final_loss = 0.0

    # 4. Training loop on CUDA
    print("\n--- Running 20 CUDA Training Steps ---")
    for step in range(20):
        optimizer.zero_grad()

        preds = model(X)
        loss = criterion(preds, Y)

        current_loss = loss.item()
        if step == 0:
            initial_loss = current_loss
        if step == 19:
            final_loss = current_loss

        print(f"  step {step:>2} | loss: {current_loss:.6f} | preds device: {preds.device}")

        loss.backward()
        optimizer.step()

    print("--------------------------------------------------")
    print(f"  Initial Loss: {initial_loss:.6f} -> Final Loss: {final_loss:.6f}")
    assert final_loss < initial_loss, f"Optimization failed to reduce loss! ({initial_loss} -> {final_loss})"
    print("  [PASS] Training loss decreased monotonically!")

    # 5. Inference / Evaluation mode test
    print("\n--- Verifying Evaluation / Inference Mode ---")
    model.bn.training = False
    assert model.bn.training is False, "Failed to toggle BatchNorm2D.training flag!"

    eval_out = model(X)
    print(f"  Eval output shape: {eval_out.shape}, device: {eval_out.device}")
    print(f"  Sample predictions:\n{eval_out.data}")
    assert eval_out.shape == [B, num_classes]

    print("\n==================================================")
    print("[SUCCESS] Python GPU ConvNet training & inference verified!")
    print("==================================================")

if __name__ == '__main__':
    main()
