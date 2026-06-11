"""
run a test to verify the cpp autograd engine with 
official pytorch autograd
"""

import subprocess
import os
import torch

def run_autograd_cpp():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    exe_path = os.path.join(project_root, "build", "autograd_main.exe")
    
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"could not find C++ exec at {exe_path}")
    
    # exec the cpp file
    result = subprocess.run([exe_path], capture_output=True, text=True, check=True)
    
    # use parsing to get data
    cpp_data={}
    for line in result.stdout.strip().split('\n'):
        if ":" in line:
            key, val = line.split(":")
            cpp_data[key.strip()] = float(val.strip())
    return cpp_data

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
    
    L = x4**2
    
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
        ("Forward Pass L", cpp.get("L"), pt["pytorch_L"]),
        ("Gradient da", cpp.get("da"), pt["pytorch_da"]),
        ("Gradient db", cpp.get("db"), pt["pytorch_db"]),
        ("Gradient dc", cpp.get("dc"), pt["pytorch_dc"]),
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
    