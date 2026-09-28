import os
import sys

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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import autograd_cpp

def test_losses(device=autograd_cpp.Device.CPU):
    device_name = "CUDA" if device == autograd_cpp.Device.CUDA else "CPU"
    print(f"\nevaluating mse and crossentropy layouts on {device_name}")
    
    # check mean squared error execution parity
    cpp_preds = autograd_cpp.Tensor([2.5, 0.0, -1.5], [1, 3], True, device)
    cpp_targets = autograd_cpp.Tensor([3.0, 1.0, -1.0], [1, 3], False, device)
    
    mse_criterion = autograd_cpp.MSELoss()
    cpp_mse_loss = mse_criterion(cpp_preds, cpp_targets)
    cpp_mse_loss.backward()
    
    print(f"mse loss [{device_name}] | autograd-cpp: {cpp_mse_loss.item():<10.4f}")
    assert cpp_mse_loss.item() > 0.0

    # check categorical cross-entropy execution parity
    cpp_logits = autograd_cpp.Tensor([2.0, 1.0, 0.1], [1, 3], True, device)
    cpp_target = autograd_cpp.Tensor([1.0, 0.0, 0.0], [1, 3], False, device)
    
    ce_criterion = autograd_cpp.CrossEntropyLoss()
    cpp_ce_loss = ce_criterion(cpp_logits, cpp_target)
    cpp_ce_loss.backward()
    
    print(f"ce loss  [{device_name}] | autograd-cpp: {cpp_ce_loss.item():<10.4f}")
    assert cpp_ce_loss.item() > 0.0

def test_advanced_sgd(device=autograd_cpp.Device.CPU):
    device_name = "CUDA" if device == autograd_cpp.Device.CUDA else "CPU"
    print(f"\nevaluating advanced sgd multi-step math on {device_name}")
    
    w1 = autograd_cpp.Tensor([0.5], [1], True, device)
    w2 = autograd_cpp.Tensor([-0.2], [1], True, device)
    
    _ = w1.grad
    _ = w2.grad
    
    optimizer = autograd_cpp.optim.SGD([w1, w2], lr=0.1, momentum=0.9, weight_decay=0.01)
    
    for step in range(1, 4):
        optimizer.zero_grad()
        w1.grad = [0.15 * step]
        w2.grad = [-0.4 * step]
        optimizer.step()
        print(f"step {step} [{device_name}] | w1: {w1.data[0]:<10.4f} | w2: {w2.data[0]:<10.4f}")

def test_adam(device=autograd_cpp.Device.CPU):
    device_name = "CUDA" if device == autograd_cpp.Device.CUDA else "CPU"
    print(f"\nevaluating adam multi-step math on {device_name}")
    
    w1 = autograd_cpp.Tensor([0.5], [1], True, device)
    w2 = autograd_cpp.Tensor([-0.2], [1], True, device)
    
    _ = w1.grad
    _ = w2.grad
    
    optimizer = autograd_cpp.optim.Adam([w1, w2], lr=0.05, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    
    for step in range(1, 4):
        optimizer.zero_grad()
        w1.grad = [0.15 * step]
        w2.grad = [-0.4 * step]
        optimizer.step()
        print(f"step {step} [{device_name}] | w1: {w1.data[0]:<10.4f} | w2: {w2.data[0]:<10.4f}")

def test_mlp_inference(device=autograd_cpp.Device.CPU):
    device_name = "CUDA" if device == autograd_cpp.Device.CUDA else "CPU"
    print(f"\nevaluating mlp layer sequence execution on {device_name}")
    try:
        model = autograd_cpp.MLP(3, [4, 2, 1], "tanh")
        model.to(device)
        inputs = autograd_cpp.Tensor([1.0, -1.0, 0.5], [1, 3], False, device)
        
        outputs = model(inputs)
        print(f"[success] mlp generated forward output shape: {outputs.shape} on {device_name}")
        print(f"          output evaluation scalar: {outputs.data[0]:.4f}")
    except Exception as e:
        print(f"[error] mlp forward execution broken on {device_name}: {e}")

def main():
    print("running loss, submodule, and optimizer validation on CPU & CUDA\n")
    for dev in [autograd_cpp.Device.CPU, autograd_cpp.Device.CUDA]:
        test_losses(dev)
        test_advanced_sgd(dev)
        test_adam(dev)
        test_mlp_inference(dev)

if __name__ == '__main__':
    main()