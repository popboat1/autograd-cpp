"""
Regression suite for C++/Python interop hardening:
- IEEE-754 NaN guards on Dropout & Dropout2D
- Sequential submodule unification (parameters, eval/train, to) & PyModule overrides
- py::keep_alive lifetime safety across gc.collect() & isolated concurrent DataLoader iterators
"""

import gc
import os
import sys

# windows mingw dll path lookup guard
for path_dir in os.environ.get("PATH", "").split(os.pathsep):
    if os.path.exists(os.path.join(path_dir, "g++.exe")) or os.path.exists(os.path.join(path_dir, "gcc.exe")):
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(path_dir)
        break

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(os.path.join(project_root, "build"))

import autograd_cpp as ag


def test_nan_guards():
    for cls in (ag.Dropout, ag.Dropout2D):
        try:
            cls(float("nan"))
            raise AssertionError(f"{cls.__name__}(nan) did not raise ValueError")
        except ValueError:
            pass
    print("[PASS] NaN probability guard raises ValueError for Dropout & Dropout2D")


def test_sequential_submodules_and_overrides():
    class MySeq(ag.Sequential):
        def __init__(self):
            super().__init__()
            self.add(ag.Conv2D(1, 2, kernel_size=3, stride=1, padding=1))
            # assign extra modules as attributes rather than via self.add()
            self.bn = ag.BatchNorm2D(2)
            self.extra = ag.Dropout(0.5)

        def forward(self, x):
            h = self.bn(super().forward(x)).relu()
            return self.extra(h)

    model = MySeq()
    # conv2d (weight+bias) + batchnorm2d (weight+bias)
    assert len(model.parameters()) == 4, f"expected 4 params, got {len(model.parameters())}"

    model.eval()
    assert model.training is False
    assert model.bn.training is False
    assert model.extra.training is False

    model.train()
    assert model.training is True
    assert model.bn.training is True
    assert model.extra.training is True

    model.to(ag.Device.CUDA)
    assert model.bn.running_mean.device == ag.Device.CUDA
    assert model.bn.running_var.device == ag.Device.CUDA
    for p in model.parameters():
        assert p.device == ag.Device.CUDA

    # verify PyModule virtual dispatch for train() and to() from C++ parent container
    override_log = []

    class CustomHookLayer(ag.Module):
        def __init__(self):
            super().__init__()
            self.lin = ag.Linear(4, 4)

        def forward(self, x):
            return self.lin(x)

        def train(self, mode=True):
            override_log.append(("train", mode))
            super().train(mode)

        def to(self, device):
            override_log.append(("to", device))
            super().to(device)

    wrapper = ag.Sequential()
    wrapper.add(CustomHookLayer())
    gc.collect()

    wrapper.eval()
    wrapper.to(ag.Device.CUDA)
    assert ("train", False) in override_log
    assert ("to", ag.Device.CUDA) in override_log

    x_inline = ag.Tensor([1.0, 2.0, 3.0, 4.0], [1, 4], False, ag.Device.CUDA)
    out = wrapper(x_inline)
    assert out.shape == [1, 4] and out.device == ag.Device.CUDA
    print("[PASS] Sequential unified submodules, PyModule overrides & keep_alive verified")


def test_dataloader_keep_alive_and_concurrent_iterators():
    class PyDS(ag.Dataset):
        def __init__(self, n):
            super().__init__()
            self.n = n

        def size(self):
            return self.n

        def get(self, index):
            x = ag.Tensor([float(index), float(index) + 0.5], [2], False)
            y = ag.Tensor([float(index)], [1], False)
            return (x, y)

    # pass inline temporary Python Dataset directly without keeping Python reference
    ag.manual_seed(123)
    dl = ag.DataLoader(PyDS(8), batch_size=2, shuffle=True, drop_last=False, device=ag.Device.CUDA)
    gc.collect()

    it1 = iter(dl)
    it2 = iter(dl)
    gc.collect()

    seen1 = []
    seen2 = []
    while True:
        done = True
        try:
            bx1, by1 = next(it1)
            assert bx1.device == ag.Device.CUDA
            seen1.extend(int(v) for v in by1.data)
            done = False
        except StopIteration:
            pass
        try:
            bx2, by2 = next(it2)
            assert bx2.device == ag.Device.CUDA
            seen2.extend(int(v) for v in by2.data)
            done = False
        except StopIteration:
            pass
        if done:
            break

    assert sorted(seen1) == list(range(8)), f"it1 corrupted: {seen1}"
    assert sorted(seen2) == list(range(8)), f"it2 corrupted: {seen2}"
    assert seen1 != seen2, "concurrent shuffled iterators unexpectedly shared identical order"
    print("[PASS] DataLoader(PyDS()) gc.collect() survival & concurrent iterator isolation verified")


if __name__ == "__main__":
    test_nan_guards()
    test_sequential_submodules_and_overrides()
    test_dataloader_keep_alive_and_concurrent_iterators()
    print("[SUCCESS] All Phase 5.3.5 Python/C++ interop regressions verified!")
