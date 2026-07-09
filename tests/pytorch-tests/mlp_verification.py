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

import torch
import autograd_cpp

def test_losses():
    print("evaluating mse and crossentropy layouts")
    
    # check mean squared error execution parity
    cpp_preds = autograd_cpp.Tensor([2.5, 0.0, -1.5], [1, 3], True)
    cpp_targets = autograd_cpp.Tensor([3.0, 1.0, -1.0], [1, 3], False)
    
    mse_criterion = autograd_cpp.MSELoss()
    cpp_mse_loss = mse_criterion(cpp_preds, cpp_targets)
    cpp_mse_loss.backward()
    
    # construct leaf node directly in 2d to preserve tracking gradients
    pt_preds = torch.tensor([[2.5, 0.0, -1.5]], requires_grad=True)
    pt_targets = torch.tensor([[3.0, 1.0, -1.0]])
    pt_mse_loss = torch.nn.functional.mse_loss(pt_preds, pt_targets)
    pt_mse_loss.backward()
    
    print(f"mse loss         | autograd-cpp: {cpp_mse_loss.data[0]:<10.4f} | pytorch: {pt_mse_loss.item():<10.4f} | {'match' if abs(cpp_mse_loss.data[0] - pt_mse_loss.item()) < 1e-5 else 'mismatch'}")
    print(f"mse grad (pred0) | autograd-cpp: {cpp_preds.grad[0]:<10.4f} | pytorch: {pt_preds.grad[0, 0].item():<10.4f} | {'match' if abs(cpp_preds.grad[0] - pt_preds.grad[0, 0].item()) < 1e-5 else 'mismatch'}")

    # check categorical cross-entropy execution parity
    cpp_logits = autograd_cpp.Tensor([2.0, 1.0, 0.1], [1, 3], True)
    # crossentropy expects one-hot target matching logit dimensions [1, 3]
    cpp_target = autograd_cpp.Tensor([1.0, 0.0, 0.0], [1, 3], False)
    
    ce_criterion = autograd_cpp.CrossEntropyLoss()
    cpp_ce_loss = ce_criterion(cpp_logits, cpp_target)
    cpp_ce_loss.backward()
    
    # construct 2d leaf logits to allow standard gradient population
    pt_logits = torch.tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    pt_target = torch.tensor([0], dtype=torch.long)
    pt_ce_loss = torch.nn.functional.cross_entropy(pt_logits, pt_target)
    pt_ce_loss.backward()
    
    print(f"ce loss          | autograd-cpp: {cpp_ce_loss.data[0]:<10.4f} | pytorch: {pt_ce_loss.item():<10.4f} | {'match' if abs(cpp_ce_loss.data[0] - pt_ce_loss.item()) < 1e-5 else 'mismatch'}")
    print(f"ce grad (logit0) | autograd-cpp: {cpp_logits.grad[0]:<10.4f} | pytorch: {pt_logits.grad[0, 0].item():<10.4f} | {'match' if abs(cpp_logits.grad[0] - pt_logits.grad[0, 0].item()) < 1e-5 else 'mismatch'}")

def test_advanced_sgd():
    print("\nevaluating advanced sgd multi-step math")
    
    # isolate separate parameters inside dedicated single element tensors
    w1 = autograd_cpp.Tensor([0.5], [1], True)
    w2 = autograd_cpp.Tensor([-0.2], [1], True)
    
    # trigger pybind property getter to force gradient vector allocation before c++ sgd initializes
    _ = w1.grad
    _ = w2.grad
    
    params_list = [w1, w2]
    
    optimizer = autograd_cpp.optim.SGD(params_list, lr=0.1, momentum=0.9, weight_decay=0.01)
    
    pt_w1 = torch.tensor([0.5], requires_grad=True)
    pt_w2 = torch.tensor([-0.2], requires_grad=True)
    pt_optimizer = torch.optim.SGD([pt_w1, pt_w2], lr=0.1, momentum=0.9, weight_decay=0.01)
    
    # execute optimization steps sequentially to evaluate momentum velocity accumulation
    for step in range(1, 3):
        optimizer.zero_grad()
        pt_optimizer.zero_grad()
        
        # inject raw gradient values directly across backend layers
        w1.grad = [0.15 * step]
        w2.grad = [-0.4 * step]
        
        pt_w1.grad = torch.tensor([0.15 * step])
        pt_w2.grad = torch.tensor([-0.4 * step])
        
        optimizer.step()
        pt_optimizer.step()
        
        print(f"step {step} param 1 data | autograd-cpp: {w1.data[0]:<10.4f} | pytorch: {pt_w1.item():<10.4f} | {'match' if abs(w1.data[0] - pt_w1.item()) < 1e-5 else 'mismatch'}")
        print(f"step {step} param 2 data | autograd-cpp: {w2.data[0]:<10.4f} | pytorch: {pt_w2.item():<10.4f} | {'match' if abs(w2.data[0] - pt_w2.item()) < 1e-5 else 'mismatch'}")

def test_mlp_inference():
    print("\nevaluating mlp layer sequence execution")
    try:
        # initialize and feed multi-dimensional tensor array batch directly to the mlp module
        model = autograd_cpp.MLP(3, [4, 2, 1], "tanh")
        inputs = autograd_cpp.Tensor([1.0, -1.0, 0.5], [1, 3], False)
        
        outputs = model.forward(inputs)
        print(f"[success] mlp successfully generated forward tensor output shape configuration: {outputs.shape}")
        print(f"          output evaluation scalar: {outputs.data[0]:.4f}")
    except Exception as e:
        print(f"[error] mlp forward execution broken: {e}")

def main():
    print("running loss, submodule, and optimizer validation\n")
    test_losses()
    test_advanced_sgd()
    test_mlp_inference()

if __name__ == '__main__':
    main()