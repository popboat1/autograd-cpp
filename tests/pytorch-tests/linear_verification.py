import os
import sys

# windows mingw dll path lookup guard
for path_dir in os.environ.get("PATH", "").split(os.pathsep):
    if os.path.exists(os.path.join(path_dir, "g++.exe")) or os.path.exists(os.path.join(path_dir, "gcc.exe")):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(path_dir)
        break

script_dir = os.path.dirname(os.path.abspath(__file__))   # framework/tests/pytorch
tests_dir = os.path.dirname(script_dir)                  # framework/tests
project_root = os.path.dirname(tests_dir)                # framework/

build_dir = os.path.join(project_root, "build")
sys.path.append(build_dir)

import torch
import autograd_cpp

def run_layer_verification():
    print("running pytorch & autograd-cpp linear layer verification...\n")
    
    fan_in = 3
    fan_out = 2

    # set global random seed for network reproducibility
    autograd_cpp.manual_seed(42)

    # instantiate native c++ structural linear layer
    linear_cpp = autograd_cpp.Linear(fan_in, fan_out, "kaiming")
    
    # define batch data input tensor layout [batch_size=1, features=3]
    x_input_vals = [1.0, -2.0, 0.5]
    x_cpp = autograd_cpp.Tensor(x_input_vals, [1, fan_in], True)
    
    # execute forward evaluation pass via c++ engine
    out_cpp = linear_cpp(x_cpp)
    
    # compute sum scalar loss to build the gradient tracking graph
    L_cpp = out_cpp.sum()
    L_cpp.backward()

    # extract weight and bias tensors from structural parameters list
    cpp_params = linear_cpp.parameters()
    cpp_w = cpp_params[0]
    cpp_b = cpp_params[1]

    # initialize matching standard reference pytorch layer
    linear_pt = torch.nn.Linear(fan_in, fan_out)
    
    # reshape flat c++ data to its true layout shape first, then transpose to match pt
    with torch.no_grad():
        linear_pt.weight.copy_(torch.tensor(cpp_w.data).reshape(fan_in, fan_out).t())
        linear_pt.bias.copy_(torch.tensor(cpp_b.data))

    # set up identical input tensor within pytorch execution stack
    x_pt = torch.tensor([x_input_vals], requires_grad=True)
    out_pt = linear_pt(x_pt)
    
    # evaluate matching evaluation scalar loss
    L_pt = out_pt.sum()
    L_pt.backward()

    # display verification evaluation report matrix
    print(f"{'metric node':<20} | {'autograd-cpp':<16} | {'pytorch':<16} | {'status'}")
    print("-" * 75)
    
    # validate forward loss value parity
    loss_match = abs(L_cpp.data[0] - L_pt.item()) < 1e-5
    status_loss = "MATCH" if loss_match else "MISMATCH"
    print(f"{'forward loss L':<20} | {L_cpp.data[0]:<16.4f} | {L_pt.item():<16.4f} | {status_loss}")
    assert loss_match, "forward pass loss mismatch!"

    # validate input gradient routing paths
    cpp_input_grads = x_cpp.grad
    pt_input_grads = x_pt.grad.flatten().tolist()
    
    for i in range(fan_in):
        grad_match = abs(cpp_input_grads[i] - pt_input_grads[i]) < 1e-5
        status_grad = "MATCH" if grad_match else "MISMATCH"
        print(f"{f'input dx[{i}] grad':<20} | {cpp_input_grads[i]:<16.4f} | {pt_input_grads[i]:<16.4f} | {status_grad}")
        assert grad_match, f"gradient routing mismatch discovered at feature axis index {i}!"

    print("\nsuccess! linear layer and parameter mapping matrix match pytorch perfectly!")

if __name__ == "__main__":
    run_layer_verification()