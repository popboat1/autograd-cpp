"""
cross-evaluates custom loss functions, submodule namespaces, 
and advanced momentum/weight-decay sgd mechanics against pytorch.
"""

import os
import sys

# windows mingw dll path lookup
for path_dir in os.environ.get("PATH", "").split(os.pathsep):
    if os.path.exists(os.path.join(path_dir, "g++.exe")) or os.path.exists(os.path.join(path_dir, "gcc.exe")):
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
    
    # mse loss check
    cpp_preds = [autograd_cpp.make_val(2.5), autograd_cpp.make_val(0.0), autograd_cpp.make_val(-1.5)]
    cpp_targets = [autograd_cpp.make_val(3.0), autograd_cpp.make_val(1.0), autograd_cpp.make_val(-1.0)]
    
    mse_criterion = autograd_cpp.MSELoss()
    cpp_mse_loss = mse_criterion(cpp_preds, cpp_targets)
    cpp_mse_loss.backward()
    
    pt_preds = torch.tensor([2.5, 0.0, -1.5], requires_grad=True)
    pt_targets = torch.tensor([3.0, 1.0, -1.0])
    pt_mse_loss = torch.nn.functional.mse_loss(pt_preds, pt_targets)
    pt_mse_loss.backward()
    
    print(f"mse loss         | autograd-cpp: {cpp_mse_loss.data:<10.4f} | pytorch: {pt_mse_loss.item():<10.4f} | {'match' if abs(cpp_mse_loss.data - pt_mse_loss.item()) < 1e-5 else 'mismatch'}")
    print(f"mse grad (pred0) | autograd-cpp: {cpp_preds[0].grad:<10.4f} | pytorch: {pt_preds.grad[0].item():<10.4f} | {'match' if abs(cpp_preds[0].grad - pt_preds.grad[0].item()) < 1e-5 else 'mismatch'}")

    # crossentropy loss check
    cpp_logits = [autograd_cpp.make_val(2.0), autograd_cpp.make_val(1.0), autograd_cpp.make_val(0.1)]
    target_idx = 0
    
    ce_criterion = autograd_cpp.CrossEntropyLoss()
    cpp_ce_loss = ce_criterion(cpp_logits, target_idx)
    cpp_ce_loss.backward()
    
    pt_logits = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
    pt_target = torch.tensor(target_idx, dtype=torch.long)
    pt_ce_loss = torch.nn.functional.cross_entropy(pt_logits.unsqueeze(0), pt_target.unsqueeze(0))
    pt_ce_loss.backward()
    
    print(f"ce loss          | autograd-cpp: {cpp_ce_loss.data:<10.4f} | pytorch: {pt_ce_loss.item():<10.4f} | {'match' if abs(cpp_ce_loss.data - pt_ce_loss.item()) < 1e-5 else 'mismatch'}")
    print(f"ce grad (logit0) | autograd-cpp: {cpp_logits[0].grad:<10.4f} | pytorch: {pt_logits.grad[0].item():<10.4f} | {'match' if abs(cpp_logits[0].grad - pt_logits.grad[0].item()) < 1e-5 else 'mismatch'}")


def test_advanced_sgd():
    print("\nevaluating advanced sgd multi-step math")
    
    # framework initializations
    w1 = autograd_cpp.make_val(0.5)
    w2 = autograd_cpp.make_val(-0.2)
    params_list = [w1, w2]
    
    # instantiate using our new optim submodule namespace
    optimizer = autograd_cpp.optim.SGD(params_list, lr=0.1, momentum=0.9, weight_decay=0.01)
    
    # pytorch matching setup
    pt_w1 = torch.tensor(0.5, requires_grad=True)
    pt_w2 = torch.tensor(-0.2, requires_grad=True)
    pt_optimizer = torch.optim.SGD([pt_w1, pt_w2], lr=0.1, momentum=0.9, weight_decay=0.01)
    
    # simulate a 2-step optimization loop sequence
    for step in range(1, 3):
        # clear out old tracking history states
        optimizer.zero_grad()
        pt_optimizer.zero_grad()
        
        # inject raw gradients directly to trace optimization adjustments
        w1.grad = 0.15 * step
        w2.grad = -0.4 * step
        
        pt_w1.grad = torch.tensor(0.15 * step)
        pt_w2.grad = torch.tensor(-0.4 * step)
        
        # trigger updates across both engines
        optimizer.step()
        pt_optimizer.step()
        
        print(f"step {step} param 1 data | autograd-cpp: {w1.data:<10.4f} | pytorch: {pt_w1.item():<10.4f} | {'match' if abs(w1.data - pt_w1.item()) < 1e-5 else 'mismatch'}")
        print(f"step {step} param 2 data | autograd-cpp: {w2.data:<10.4f} | pytorch: {pt_w2.item():<10.4f} | {'match' if abs(w2.data - pt_w2.item()) < 1e-5 else 'mismatch'}")


def test_mlp_inference():
    print("\nevaluating mlp layer sequence execution")
    try:
        model = autograd_cpp.MLP(3, [4, 2, 1], "tanh", 42)
        inputs = [autograd_cpp.make_val(1.0), autograd_cpp.make_val(-1.0), autograd_cpp.make_val(0.5)]
        
        outputs = model.forward(inputs)
        print(f"[success] mlp successfully generated forward vector prediction size: {len(outputs)}")
        print(f"          output raw data scalar: {outputs[0].data:.4f}")
    except Exception as e:
        print(f"[error] mlp forward execution broken: {e}")


def main():
    print("running loss, submodule, and optimizer validation\n")
    test_losses()
    test_advanced_sgd()
    test_mlp_inference()

if __name__ == '__main__':
    main()