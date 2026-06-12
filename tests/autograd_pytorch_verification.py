"""
run a test to verify the cpp autograd engine with 
official pytorch autograd
"""

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

def run_autograd_cpp():
    a = autograd_cpp.make_val(-2.0)
    b = autograd_cpp.make_val(3.0)
    c = autograd_cpp.make_val(10.0)
    
    # forward pass (same as main.cpp)
    x1 = a * b
    x2 = x1 + c
    x3 = x2 - a
    x4 = x3 / b
    
    # activations
    x5 = x4.tanh()
    x6 = x5.exp()
    x7 = x6.relu()
    
    L = x7**2
    
    L.backward()
    
    return {
        "L": L.data,
        "da": a.grad,
        "db": b.grad,
        "dc": c.grad
    }

def run_pytorch():
    a = torch.tensor(-2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    c = torch.tensor(10.0, requires_grad=True)
    
    # forward pass (same as main.cpp)
    x1 = a * b
    x2 = x1 + c
    x3 = x2 - a
    x4 = x3 / b
    
    # activations
    x5 = torch.tanh(x4)
    x6 = torch.exp(x5)
    x7 = torch.relu(x6)
    
    L = x7**2
    
    L.backward()
    
    return {
        "pytorch_L": L.item(),
        "pytorch_da": a.grad.item(),
        "pytorch_db": b.grad.item(),
        "pytorch_dc": c.grad.item()
    }
    
def main():
    print("running pytorch & autograd-cpp verification...")
    
    try:
        cpp = run_autograd_cpp()
        pt = run_pytorch()
    except Exception as e:
        print(f"[ERROR] execution failed: {e}")
        return
    
    # print side by side comparison
    print(f"{'Metric':<15} | {"c++":<16} | {'pytorch':<16} | {'status'}")
    mappings = [
        ("Forward Pass L", cpp["L"], pt["pytorch_L"]),
        ("Gradient da", cpp["da"], pt["pytorch_da"]),
        ("Gradient db", cpp["db"], pt["pytorch_db"]),
        ("Gradient dc", cpp["dc"], pt["pytorch_dc"]),
    ]
    
    all_passed = True
    for metric, cpp_val, pt_val in mappings:
        if cpp_val is None:
            print(f"[ERROR] missing data for {metric}")
            all_passed = False
            continue
        
        match = abs(cpp_val - pt_val) < 1e-5
        status = "MATCH" if match else "MISMATCH"
        if not match:
            all_passed = False
            
        print(f"{metric:<15} | {cpp_val:<16.4f} | {pt_val:<16.4f} | {status}")
    
    if all_passed:
        print("[SUCCESS] the autograd c++ engine matches pytorch!1!!")
    else:
        print("[ERROR] gradient mismatched")
    
if __name__ == '__main__':
    main()
    