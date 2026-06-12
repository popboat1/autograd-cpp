import os
import sys

for path_dir in os.environ.get("PATH", "").split(os.pathsep):
    if os.path.exists(os.path.join(path_dir, "g++.exe")) or os.path.exists(os.path.join(path_dir, "gcc.exe")):
        os.add_dll_directory(path_dir)
        break

script_dir = os.path.dirname(os.path.abspath(__file__))  # framework/tests
project_root = os.path.dirname(script_dir)              # framework/
build_dir = os.path.join(project_root, "build")         # framework/build
sys.path.append(build_dir)

import torch
import autograd_cpp

def run_layer_verification():
    print("running pytorch & autograd-cpp linear layer verification...\n")
    
    fan_in = 3
    fan_out = 2

    # run cpp backend
    linear_cpp = autograd_cpp.Linear(fan_in, fan_out, 42)# seed 42 for reproducibility
    
    x_input_vals = [1.0, -2.0, 0.5]
    x_cpp = [autograd_cpp.make_val(v) for v in x_input_vals]
    
    # forward pass through cpp engine
    out_cpp = linear_cpp.forward(x_cpp)
    
    # calculate loss
    L_cpp = out_cpp[0] + out_cpp[1]
    L_cpp.backward()

    # extract weights and biases
    cpp_params = linear_cpp.parameters() # flat list of 8 Value items

    # ----------------------------------------------------
    # pytorch run
    linear_pt = torch.nn.Linear(fan_in, fan_out)
    
    # dynamically inject the exact values from C++ memory into PyTorch
    with torch.no_grad():
        idx = 0
        for i in range(fan_out):
            for j in range(fan_in):
                linear_pt.weight[i, j] = cpp_params[idx].data
                idx += 1
        for i in range(fan_out):
            linear_pt.bias[i] = cpp_params[idx].data
            idx += 1

    # pytorch input setup
    x_pt = torch.tensor(x_input_vals, requires_grad=True)
    out_pt = linear_pt(x_pt)
    
    # matching loss in pytorch
    L_pt = out_pt[0] + out_pt[1]
    L_pt.backward()

    # compare both results
    print(f"{'metric node':<20} | {'autograd-cpp':<16} | {'pytorcj':<16} | {'status'}")
    print("-" * 75)
    
    # output loss comparison
    loss_match = abs(L_cpp.data - L_pt.item()) < 1e-5
    status_loss = "MATCH" if loss_match else "MISMATCH"
    print(f"{'forward loss L':<20} | {L_cpp.data:<16.4f} | {L_pt.item():<16.4f} | {status_loss}")
    assert loss_match, "forward pass loss mismatch!"

    # gradients comparison
    for i in range(fan_in):
        cpp_grad = x_cpp[i].grad
        pt_grad = x_pt.grad[i].item()
        grad_match = abs(cpp_grad - pt_grad) < 1e-5
        status_grad = "MATCH" if grad_match else "MISMATCH"
        print(f"{f'input dx[{i}] grad':<20} | {cpp_grad:<16.4f} | {pt_grad:<16.4f} | {status_grad}")
        assert grad_match, f"Gradient mismatch at input variable index {i}!"

    print("SUCCESS! Linear layer and parameter mapping matrix match PyTorch perfectly!1!!")

if __name__ == "__main__":
    run_layer_verification()