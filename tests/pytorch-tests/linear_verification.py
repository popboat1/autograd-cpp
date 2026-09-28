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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import autograd_cpp

def run_layer_verification():
    print("running autograd-cpp linear layer CPU & CUDA verification...\n")
    
    fan_in = 3
    fan_out = 2
    x_input_vals = [1.0, -2.0, 0.5]

    for device_name, device in [("CPU", autograd_cpp.Device.CPU), ("CUDA", autograd_cpp.Device.CUDA)]:
        print(f"\n{'=' * 20} Testing on {device_name} {'=' * 20}")

        # set global random seed for network reproducibility
        autograd_cpp.manual_seed(42)

        # instantiate native c++ structural linear layer
        linear_cpp = autograd_cpp.Linear(fan_in, fan_out, "kaiming")
        linear_cpp.to(device)
        
        # define batch data input tensor layout [batch_size=1, features=3]
        x_cpp = autograd_cpp.Tensor(x_input_vals, [1, fan_in], True, device)
        
        # execute forward evaluation pass via c++ engine
        out_cpp = linear_cpp(x_cpp)
        
        # compute sum scalar loss to build the gradient tracking graph
        L_cpp = out_cpp.sum()
        L_cpp.backward()

        # extract weight and bias tensors from structural parameters list
        cpp_w = linear_cpp.weights
        cpp_b = linear_cpp.biases

        if HAS_TORCH:
            # initialize matching standard reference pytorch layer
            linear_pt = torch.nn.Linear(fan_in, fan_out)
            
            with torch.no_grad():
                linear_pt.weight.copy_(torch.tensor(list(cpp_w.data)).reshape(fan_in, fan_out).t())
                linear_pt.bias.copy_(torch.tensor(list(cpp_b.data)))

            x_pt = torch.tensor([x_input_vals], requires_grad=True)
            out_pt = linear_pt(x_pt)
            
            L_pt = out_pt.sum()
            L_pt.backward()

            # display verification evaluation report matrix
            print(f"{'metric node':<20} | {'autograd-cpp':<16} | {'pytorch':<16} | {'status'}")
            print("-" * 75)
            
            loss_match = abs(L_cpp.item() - L_pt.item()) < 1e-5
            status_loss = "MATCH" if loss_match else "MISMATCH"
            print(f"{'forward loss L':<20} | {L_cpp.item():<16.4f} | {L_pt.item():<16.4f} | {status_loss}")
            assert loss_match, f"forward pass loss mismatch on {device_name}!"

            cpp_input_grads = list(x_cpp.grad)
            pt_input_grads = x_pt.grad.flatten().tolist()
            
            for i in range(fan_in):
                grad_match = abs(cpp_input_grads[i] - pt_input_grads[i]) < 1e-5
                status_grad = "MATCH" if grad_match else "MISMATCH"
                print(f"{f'input dx[{i}] grad':<20} | {cpp_input_grads[i]:<16.4f} | {pt_input_grads[i]:<16.4f} | {status_grad}")
                assert grad_match, f"gradient routing mismatch discovered at feature axis index {i} on {device_name}!"
        else:
            print(f"loss: {L_cpp.item():.4f}")
            print(f"input gradients: {list(x_cpp.grad)}")
            print(f"weight gradients: {list(cpp_w.grad)}")
            print(f"bias gradients: {list(cpp_b.grad)}")
            assert abs(L_cpp.item()) > 0.0, "Loss evaluated to zero!"

    print("\nsuccess! linear layer and parameter mapping matrix verified cleanly on CPU & CUDA!")

if __name__ == "__main__":
    run_layer_verification()