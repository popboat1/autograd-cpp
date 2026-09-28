"""
run a test to verify the cpp autograd engine with 
official pytorch autograd and verify CPU / CUDA parity
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

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import autograd_cpp

# baseline numeric initialization constants
A_vals = [0.5, -0.2, 0.1, 0.8, 0.3, -0.5]  # shape: (2, 3)
B_vals = [0.2, 0.7, -0.4, 0.1, 0.6, -0.3]  # shape: (3, 2)
C_vals = [0.1, 0.2, 0.3, 0.4]              # shape: (1, 4)
D_vals = [2.0, 2.0, 2.0, 2.0]              # shape: (1, 4)
E_vals = [0.5, 0.5, 0.5, 0.5]              # shape: (1, 4)
F_vals = [2.0, 2.0, 2.0, 2.0]              # shape: (1, 4)

def run_autograd_cpp(device=autograd_cpp.Device.CPU):
    # instantiate inputs as formal framework multi-dimensional tensors
    A = autograd_cpp.Tensor(A_vals, [2, 3], True, device)
    B = autograd_cpp.Tensor(B_vals, [3, 2], True, device)
    C = autograd_cpp.Tensor(C_vals, [1, 4], True, device)
    D = autograd_cpp.Tensor(D_vals, [1, 4], False, device)
    E = autograd_cpp.Tensor(E_vals, [1, 4], False, device)
    F = autograd_cpp.Tensor(F_vals, [1, 4], False, device)
    
    # structural matrix transformations
    x1 = A @ B                        # matmul -> (2, 2)
    x2 = x1.permute([1, 0])           # permute -> (2, 2) [non-contiguous view]
    x3 = x2.reshape([4, 1])           # reshape -> (4, 1)
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
    
    # Extra evaluation views checking index reductions safely
    _ = x11.max(1, False)
    _ = x11.argmax(1, False)
    
    # Final global flat reduction scalar setup
    Loss = x13.sum()
    Loss.backward()
    
    return {
        "forward_out": list(Loss.data),
        "grad_A": list(A.grad),
        "grad_B": list(B.grad),
        "grad_C": list(C.grad)
    }

def run_pytorch():
    if not HAS_TORCH:
        return None
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
    
def verify_and_print(title, test_list, ref_list, test_name="autograd_cpp", ref_name="pytorch", tol=1e-5):
    print(f"\n=== Verifying Layout: {title} ===")
    print(f"{'Index':<8} | {test_name:<16} | {ref_name:<16} | {'status'}")
    print("-" * 60)
    
    if len(test_list) != len(ref_list):
        print(f"[ERROR] Dimensional element count mismatch: {len(test_list)} vs {len(ref_list)}")
        return False
        
    passed = True
    for i, (test_v, ref_v) in enumerate(zip(test_list, ref_list)):
        match = math.isclose(test_v, ref_v, abs_tol=tol)
        status = "MATCH" if match else "MISMATCH"
        if not match:
            passed = False
        print(f"{i:<8} | {test_v:<16.6f} | {ref_v:<16.6f} | {status}")
    return passed

def main():
    print("Initializing comprehensive framework cross-verification suite...")
    
    try:
        cpu_results = run_autograd_cpp(autograd_cpp.Device.CPU)
        gpu_results = run_autograd_cpp(autograd_cpp.Device.CUDA)
        pt_results = run_pytorch()
    except Exception as e:
        print(f"[CRITICAL ERROR] Graph computation execution broken: {e}")
        import traceback
        traceback.print_exc()
        return

    # 1. CPU vs GPU Parity Verification
    print("\n" + "=" * 60)
    print("STEP 1: Verifying autograd-cpp CPU vs CUDA Parity")
    print("=" * 60)
    checks_gpu = [
        ("Loss Forward Output Scalar (GPU vs CPU)", gpu_results["forward_out"], cpu_results["forward_out"]),
        ("Gradient Matrix dL/dA (GPU vs CPU)", gpu_results["grad_A"], cpu_results["grad_A"]),
        ("Gradient Matrix dL/dB (GPU vs CPU)", gpu_results["grad_B"], cpu_results["grad_B"]),
        ("Gradient Vector dL/dC (GPU vs CPU)", gpu_results["grad_C"], cpu_results["grad_C"]),
    ]
    
    gpu_success = True
    for title, gpu_arr, cpu_arr in checks_gpu:
        if not verify_and_print(title, gpu_arr, cpu_arr, "autograd_GPU", "autograd_CPU"):
            gpu_success = False

    # 2. PyTorch Parity Verification (if torch is installed)
    pt_success = True
    if pt_results is not None:
        print("\n" + "=" * 60)
        print("STEP 2: Verifying autograd-cpp GPU vs PyTorch Baseline")
        print("=" * 60)
        checks_pt = [
            ("Loss Forward Output Scalar", gpu_results["forward_out"], pt_results["forward_out"]),
            ("Gradient Matrix dL/dA", gpu_results["grad_A"], pt_results["grad_A"]),
            ("Gradient Matrix dL/dB", gpu_results["grad_B"], pt_results["grad_B"]),
            ("Gradient Vector dL/dC", gpu_results["grad_C"], pt_results["grad_C"]),
        ]
        for title, gpu_arr, pt_arr in checks_pt:
            if not verify_and_print(title, gpu_arr, pt_arr, "autograd_GPU", "PyTorch"):
                pt_success = False
    else:
        print("\n[NOTE] PyTorch not installed in active environment - skipped PyTorch comparison.")

    print("\n" + "=" * 60)
    if gpu_success and pt_success:
        print("[SUCCESS] All multi-dimensional math, layout views, and activations match cleanly!")
    else:
        print("[FAILURE] Gradient misalignment detected.")
    print("=" * 60)

if __name__ == '__main__':
    main()