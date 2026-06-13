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

script_dir = os.path.dirname(os.path.abspath(__file__))   # framework/tests/pytorch
tests_dir = os.path.dirname(script_dir)                  # framework/tests
project_root = os.path.dirname(tests_dir)                # framework/

build_dir = os.path.join(project_root, "build")
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
    
def run_requires_grad_cpp():
    a = autograd_cpp.make_val(2.0, True)
    b = autograd_cpp.make_val(3.0, False)
    
    c = b * b
    Loss = a + c
    Loss.backward()
    
    return {
        "Loss": Loss.data,
        "da": a.grad,
        "db": b.grad
    }

def run_requires_grad_pytorch():
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=False)
    
    c = b * b
    Loss = a + c
    Loss.backward()
    
    b_grad_val = 0.0 if b.grad is None else b.grad.item()
    
    return {
        "pytorch_Loss": Loss.item(),
        "pytorch_da": a.grad.item(),
        "pytorch_db": b_grad_val
    }
    
def print_table(title, mappings):
    print(f"\n--- {title} ---")
    print(f"{'metric Node':<20} | {'autograd-cpp':<16} | {'pytorch':<16} | {'status'}")
    print("-" * 75)
    
    all_passed = True
    for metric, cpp_val, pt_val in mappings:
        if cpp_val is None:
            print(f"[ERROR] missing data tracking block for: {metric}")
            all_passed = False
            continue
        
        match = abs(cpp_val - pt_val) < 1e-5
        status = "MATCH" if match else "MISMATCH"
        if not match:
            all_passed = False
            
        print(f"{metric:<20} | {cpp_val:<16.4f} | {pt_val:<16.4f} | {status}")
    return all_passed
    
def main():
    print("running pytorch & autograd-cpp verification...")
    
    try:
        cpp = run_autograd_cpp()
        pt = run_pytorch()
        
        cpp_freeze = run_requires_grad_cpp()
        pt_freeze = run_requires_grad_pytorch()
        
    except Exception as e:
        print(f"[ERROR] execution failed: {e}")
        return
    
    # print side by side comparison
    math_mappings = [
        ("Forward Pass L", cpp["L"], pt["pytorch_L"]),
        ("Gradient da", cpp["da"], pt["pytorch_da"]),
        ("Gradient db", cpp["db"], pt["pytorch_db"]),
        ("Gradient dc", cpp["dc"], pt["pytorch_dc"]),
    ]
    
    freeze_mappings = [
        ("Mixed Pass Loss", cpp_freeze["Loss"], pt_freeze["pytorch_Loss"]),
        ("Tracking Node da", cpp_freeze["da"], pt_freeze["pytorch_da"]),
        ("Frozen Node db", cpp_freeze["db"], pt_freeze["pytorch_db"]),
    ]
    
    math_success = print_table("core math graphs", math_mappings)
    freeze_success = print_table("graphs freezing", freeze_mappings)
    
    if math_success and freeze_success:
        print("[SUCCESS] the autograd c++ engine matches pytorch!1!!")
    else:
        print("[ERROR] gradient mismatched")
    
if __name__ == '__main__':
    main()
    