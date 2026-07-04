"""
run a test to verify the cpp autograd engine with 
official pytorch autograd
"""

import os
import sys
import math

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

# baseline numeric initialization constants
A_vals = [0.5, -0.2, 0.1, 0.8, 0.3, -0.5]  # shape: (2, 3)
B_vals = [0.2, 0.7, -0.4, 0.1, 0.6, -0.3]  # shape: (3, 2)
C_vals = [0.1, 0.2, 0.3, 0.4]              # shape: (1, 4)
D_vals = [2.0, 2.0, 2.0, 2.0]              # shape: (1, 4)
E_vals = [0.5, 0.5, 0.5, 0.5]              # shape: (1, 4)
F_vals = [2.0, 2.0, 2.0, 2.0]              # shape: (1, 4)

def run_autograd_cpp():
    # instantiate inputs as formal framework multi-dimensional tensors
    A = autograd_cpp.Tensor(A_vals, [2, 3], True)
    B = autograd_cpp.Tensor(B_vals, [3, 2], True)
    C = autograd_cpp.Tensor(C_vals, [1, 4], True)
    D = autograd_cpp.Tensor(D_vals, [1, 4], False)
    E = autograd_cpp.Tensor(E_vals, [1, 4], False)
    F = autograd_cpp.Tensor(F_vals, [1, 4], False)
    
    # structural matrix transformations
    x1 = A @ B                        # matmul -> (2, 2)
    x2 = x1.permute([1, 0])           # permute -> (2, 2) [non-contiguous view]
    x3 = x2.reshape([4, 1])           # reshape -> (4, 1) [forces contiguity copy]
    x4 = x3.squeeze(1)                # squeeze -> (4,)
    x5 = x4.unsqueeze(0)              # unsqueeze -> (1, 4)
    
    # Element-wise arithmetic tensor interactions
    x6 = x5 + C                       # Addition
    x7 = x6 * D                       # Multiplication
    x8 = x7 - E                       # Subtraction
    x9 = x8 / F                       # Division
    x10 = x9.pow(2.0)                 # Power scalar scaling
    
    # Non-linear mathematical activations sequence
    x11 = x10.relu().tanh().exp().sigmoid().log()
    
    # Multi-dimensional structural reductions
    x12 = x11.sum(1, True)            # Dimensional sum reduction -> (1, 1)
    x13 = x12.mean(0, False)          # Dimensional mean reduction -> (1,)
    
    # Extra evaluation views checking index reductions safely (untracked)
    _ = x11.max(1, False)
    _ = x11.argmax(1, False)
    
    # Final global flat reduction scalar setup
    Loss = x13.sum()
    Loss.backward()
    
    return {
        "forward_out": Loss.data,
        "grad_A": A.grad,
        "grad_B": B.grad,
        "grad_C": C.grad
    }

def run_pytorch():
    # Mirror explicit execution graph setups natively inside torch
    A = torch.tensor(A_vals, dtype=torch.float64).reshape(2, 3).clone().detach().requires_grad_(True)
    B = torch.tensor(B_vals, dtype=torch.float64).reshape(3, 2).clone().detach().requires_grad_(True)
    C = torch.tensor(C_vals, dtype=torch.float64).reshape(1, 4).clone().detach().requires_grad_(True)
    D = torch.tensor(D_vals, dtype=torch.float64).reshape(1, 4)
    E = torch.tensor(E_vals, dtype=torch.float64).reshape(1, 4)
    F = torch.tensor(F_vals, dtype=torch.float64).reshape(1, 4)
    
    x1 = A @ B
    x2 = x1.permute(1, 0)
    x3 = x2.reshape(4, 1)
    x4 = x3.squeeze(1)
    x5 = x4.unsqueeze(0)
    
    x6 = x5 + C
    x7 = x6 * D
    x8 = x7 - E
    x9 = x8 / F
    x10 = x9.pow(2.0)
    
    x11 = torch.log(torch.sigmoid(torch.exp(torch.tanh(torch.relu(x10)))))
    
    x12 = x11.sum(dim=1, keepdim=True)
    x13 = x12.mean(dim=0, keepdim=False)
    
    Loss = x13.sum()
    Loss.backward()
    
    return {
        "forward_out": [Loss.item()],
        "grad_A": A.grad.flatten().tolist(),
        "grad_B": B.grad.flatten().tolist(),
        "grad_C": C.grad.flatten().tolist()
    }
    
def verify_and_print(title, cpp_list, pt_list, tol=1e-5):
    print(f"\n=== Verifying Layout: {title} ===")
    print(f"{'Index':<8} | {'autograd_cpp':<16} | {'pytorch':<16} | {'status'}")
    print("-" * 60)
    
    if len(cpp_list) != len(pt_list):
        print(f"[ERROR] Dimensional element count mismatch: {len(cpp_list)} vs {len(pt_list)}")
        return False
        
    passed = True
    for i, (cpp_v, pt_v) in enumerate(zip(cpp_list, pt_list)):
        match = math.isclose(cpp_v, pt_v, abs_tol=tol)
        status = "MATCH" if match else "MISMATCH"
        if not match:
            passed = False
        print(f"{i:<8} | {cpp_v:<16.6f} | {pt_v:<16.6f} | {status}")
    return passed

def main():
    print("Initializing comprehensive framework cross-verification suite...")
    
    try:
        cpp_results = run_autograd_cpp()
        pt_results = run_pytorch()
    except Exception as e:
        print(f"[CRITICAL ERROR] Graph computation execution broken: {e}")
        import traceback
        traceback.print_exc()
        return

    # Aggregate metric alignment runs
    checks = [
        ("Loss Forward Output Scalar", cpp_results["forward_out"], pt_results["forward_out"]),
        ("Gradient Matrix dL/dA", cpp_results["grad_A"], pt_results["grad_A"]),
        ("Gradient Matrix dL/dB", cpp_results["grad_B"], pt_results["grad_B"]),
        ("Gradient Vector dL/dC", cpp_results["grad_C"], pt_results["grad_C"]),
    ]
    
    global_success = True
    for title, cpp_arr, pt_arr in checks:
        if not verify_and_print(title, cpp_arr, pt_arr):
            global_success = False
            
    print("\n" + "=" * 60)
    if global_success:
        print("[SUCCESS] All multi-dimensional math, layout views, and activations match PyTorch perfectly!")
    else:
        print("[FAILURE] Gradient misalignment detected between the autograd representations.")
    print("=" * 60)

if __name__ == '__main__':
    main()