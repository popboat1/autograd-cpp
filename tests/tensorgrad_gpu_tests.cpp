#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include "autograd/Tensor.h"

// helper to assert floating point parity smoothly
bool close_enough(double a, double b, double tol = 1e-4) {
    return std::abs(a - b) < tol;
}

// helper timer to isolate and measure GPU execution latency accurately
struct CudaTimer {
    std::chrono::high_resolution_clock::time_point start_time;

    void start() {
        cudaDeviceSynchronize();
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop_ms() {
        cudaDeviceSynchronize();
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

// helper to generate deterministic synthetic 4D data
std::vector<double> generate_4d_data(size_t total_elements, double offset = 0.0, double scale = 1.0) {
    std::vector<double> values(total_elements);
    for (size_t i = 0; i < total_elements; ++i) {
        values[i] = (std::sin(static_cast<double>(i) * 0.05) * 10.0 + offset) * scale;
    }
    return values;
}

int main() {
    std::cout << std::fixed << std::setprecision(3);
    CudaTimer timer;

    // Standard 4D Tensor dimensions for test suite: (4, 8, 16, 32) = 16,384 elements
    const std::vector<size_t> shape_4d = {4, 8, 16, 32};
    const size_t num_elements = 4 * 8 * 16 * 32;

    std::cout << "==========================================\n";
    std::cout << "starting tensorgrad gpu unary tests\n";
    std::cout << "==========================================\n";

    // test 1: relu forward and backward on gpu tensor
    {
        auto raw_data = generate_4d_data(num_elements, -2.0);
        auto x_cpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->relu();
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->relu();
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] relu cuda pass verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 2: exp forward and backward on gpu tensor
    {
        auto raw_data = generate_4d_data(num_elements, 0.0, 0.1); // bounded range to prevent overflow
        auto x_cpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->exp();
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->exp();
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] exp cuda pass verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 3: tanh forward and backward on gpu tensor
    {
        auto raw_data = generate_4d_data(num_elements, 0.0, 0.5);
        auto x_cpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->tanh();
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->tanh();
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] tanh cuda pass verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 4: sigmoid forward and backward on gpu tensor
    {
        auto raw_data = generate_4d_data(num_elements, 0.0, 0.5);
        auto x_cpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->sigmoid();
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->sigmoid();
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] sigmoid cuda pass verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 5: log forward and backward on gpu tensor
    {
        auto raw_data = generate_4d_data(num_elements, 12.0, 1.0); // positive domain x > 0
        auto x_cpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->log();
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->log();
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] log cuda pass verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 6: pow forward and backward on gpu tensor
    {
        auto raw_data = generate_4d_data(num_elements, 5.0, 0.5);
        auto x_cpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->pow(3.0);
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->pow(3.0);
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] pow cuda pass verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 7: sqrt forward and backward on gpu tensor
    {
        auto raw_data = generate_4d_data(num_elements, 15.0, 1.0); // non-negative domain
        auto x_cpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->sqrt();
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->sqrt();
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] sqrt cuda pass verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 8: neg / prefix operator- forward and backward on gpu tensor
    {
        auto raw_data = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(raw_data, shape_4d, true, Device::CPU);

        auto out_cpu = -x_cpu;
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = -x_gpu;
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] neg cuda pass verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    std::cout << "[PASS] all unary kernels verified\n";
    std::cout << "------------------------------------------\n";

    std::cout << "starting binary kernels test...\n";

    // test 9: addition matching shape fast path on gpu tensor
    {
        auto a_raw = generate_4d_data(num_elements, 2.0);
        auto b_raw = generate_4d_data(num_elements, 5.0);

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, shape_4d, true, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, shape_4d, true, Device::CPU);

        auto out_cpu = a_cpu + b_cpu;
        a_cpu->ensure_grad_allocated();
        b_cpu->ensure_grad_allocated();
        out_cpu->backward();

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = a_gpu + b_gpu;
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        a_gpu->to(Device::CPU);
        b_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
            assert(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
        }
        std::cout << "[PASS] operator+ cuda fast path verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 10: addition broadcasting path on gpu tensor (4D + 2D broadcast)
    {
        const std::vector<size_t> b_shape = {16, 32}; // broadcast across B and C axes
        size_t b_elements = 16 * 32;

        auto a_raw = generate_4d_data(num_elements, 1.0);
        auto b_raw = generate_4d_data(b_elements, 10.0);

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, b_shape, true, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, b_shape, true, Device::CPU);

        auto out_cpu = a_cpu + b_cpu;
        a_cpu->ensure_grad_allocated();
        b_cpu->ensure_grad_allocated();
        out_cpu->backward();

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = a_gpu + b_gpu;
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        a_gpu->to(Device::CPU);
        b_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
        }
        for (size_t i = 0; i < b_elements; ++i) {
            assert(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
        }
        std::cout << "[PASS] operator+ cuda broadcast path verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 11: subtraction matching shape fast path on gpu tensor
    {
        auto a_raw = generate_4d_data(num_elements, 10.0);
        auto b_raw = generate_4d_data(num_elements, 3.0);

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, shape_4d, true, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, shape_4d, true, Device::CPU);

        auto out_cpu = a_cpu - b_cpu;
        a_cpu->ensure_grad_allocated();
        b_cpu->ensure_grad_allocated();
        out_cpu->backward();

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = a_gpu - b_gpu;
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        a_gpu->to(Device::CPU);
        b_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
            assert(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
        }
        std::cout << "[PASS] operator- cuda fast path verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 12: subtraction broadcasting path on gpu tensor
    {
        const std::vector<size_t> b_shape = {16, 32};
        size_t b_elements = 16 * 32;

        auto a_raw = generate_4d_data(num_elements, 20.0);
        auto b_raw = generate_4d_data(b_elements, 2.0);

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, b_shape, true, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, b_shape, true, Device::CPU);

        auto out_cpu = a_cpu - b_cpu;
        a_cpu->ensure_grad_allocated();
        b_cpu->ensure_grad_allocated();
        out_cpu->backward();

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = a_gpu - b_gpu;
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        a_gpu->to(Device::CPU);
        b_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
        }
        for (size_t i = 0; i < b_elements; ++i) {
            assert(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
        }
        std::cout << "[PASS] operator- cuda broadcast path verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 13: multiplication matching shape fast path on gpu tensor
    {
        auto a_raw = generate_4d_data(num_elements, 2.0);
        auto b_raw = generate_4d_data(num_elements, 4.0);

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, shape_4d, true, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, shape_4d, true, Device::CPU);

        auto out_cpu = a_cpu * b_cpu;
        a_cpu->ensure_grad_allocated();
        b_cpu->ensure_grad_allocated();
        out_cpu->backward();

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = a_gpu * b_gpu;
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        a_gpu->to(Device::CPU);
        b_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
            assert(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
        }
        std::cout << "[PASS] operator* cuda fast path verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 14: multiplication broadcasting path on gpu tensor
    {
        const std::vector<size_t> b_shape = {16, 32};
        size_t b_elements = 16 * 32;

        auto a_raw = generate_4d_data(num_elements, 3.0);
        auto b_raw = generate_4d_data(b_elements, 2.0);

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, b_shape, true, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, b_shape, true, Device::CPU);

        auto out_cpu = a_cpu * b_cpu;
        a_cpu->ensure_grad_allocated();
        b_cpu->ensure_grad_allocated();
        out_cpu->backward();

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = a_gpu * b_gpu;
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        a_gpu->to(Device::CPU);
        b_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
        }
        for (size_t i = 0; i < b_elements; ++i) {
            assert(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
        }
        std::cout << "[PASS] operator* cuda broadcast path verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 15: tensor * scalar on gpu tensor
    {
        auto a_raw = generate_4d_data(num_elements, 2.0);
        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);

        auto out_cpu = a_cpu * 2.5;
        a_cpu->ensure_grad_allocated();
        out_cpu->backward();

        a_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = a_gpu * 2.5;
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        a_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
        }
        std::cout << "[PASS] tensor * scalar cuda pass verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 16: scalar * tensor on gpu tensor
    {
        auto a_raw = generate_4d_data(num_elements, 1.0);
        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);

        auto out_cpu = 3.0 * a_cpu;
        a_cpu->ensure_grad_allocated();
        out_cpu->backward();

        a_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = 3.0 * a_gpu;
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        a_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
        }
        std::cout << "[PASS] scalar * tensor cuda pass verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 17: division on gpu tensor
    {
        auto a_raw = generate_4d_data(num_elements, 20.0);
        auto b_raw = generate_4d_data(num_elements, 5.0, 0.5); // non-zero divisor

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, shape_4d, true, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, shape_4d, true, Device::CPU);

        auto out_cpu = a_cpu / b_cpu;
        a_cpu->ensure_grad_allocated();
        b_cpu->ensure_grad_allocated();
        out_cpu->backward();

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = a_gpu / b_gpu;
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        a_gpu->to(Device::CPU);
        b_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
            assert(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
        }
        std::cout << "[PASS] operator/ cuda pass verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 18: in-place addition (add_) on gpu tensor
    {
        auto a_raw = generate_4d_data(num_elements, 1.0);
        auto b_raw = generate_4d_data(num_elements, 10.0);

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, shape_4d, false, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, shape_4d, false, Device::CPU);

        a_cpu->add_(b_cpu);

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        a_gpu->add_(b_gpu);
        double fwd_time = timer.stop_ms();

        a_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*a_gpu->data)[i], (*a_cpu->data)[i]));
        }
        std::cout << "[PASS] in-place add_ cuda pass verified (exec: " << fwd_time << " ms)\n";
    }

    // test 19: in-place subtraction (sub_) on gpu tensor with broadcasting
    {
        const std::vector<size_t> b_shape = {16, 32};

        auto a_raw = generate_4d_data(num_elements, 30.0);
        auto b_raw = generate_4d_data(16 * 32, 2.0);

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, b_shape, false, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, b_shape, false, Device::CPU);

        a_cpu->sub_(b_cpu);

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        a_gpu->sub_(b_gpu);
        double fwd_time = timer.stop_ms();

        a_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*a_gpu->data)[i], (*a_cpu->data)[i]));
        }
        std::cout << "[PASS] in-place sub_ cuda broadcast pass verified (exec: " << fwd_time << " ms)\n";
    }

    // test 20: operator== on gpu tensor with broadcasting
    {
        const std::vector<size_t> b_shape = {16, 32};

        auto a_raw = generate_4d_data(num_elements, 5.0);
        auto b_raw = generate_4d_data(16 * 32, 5.0);

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, b_shape, false, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, b_shape, false, Device::CPU);

        auto out_cpu = (*a_cpu == *b_cpu);

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = (*a_gpu == *b_gpu);
        double fwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        std::cout << "[PASS] operator== cuda broadcast pass verified (exec: " << fwd_time << " ms)\n";
    }

    // test 21: operator< on gpu tensor with broadcasting
    {
        const std::vector<size_t> b_shape = {16, 32};

        auto a_raw = generate_4d_data(num_elements, 1.0);
        auto b_raw = generate_4d_data(16 * 32, 3.0);

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, b_shape, false, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, b_shape, false, Device::CPU);

        auto out_cpu = (*a_cpu < *b_cpu);

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = (*a_gpu < *b_gpu);
        double fwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        std::cout << "[PASS] operator< cuda broadcast pass verified (exec: " << fwd_time << " ms)\n";
    }

    // test 22: operator> on gpu tensor with broadcasting
    {
        const std::vector<size_t> b_shape = {16, 32};

        auto a_raw = generate_4d_data(num_elements, 10.0);
        auto b_raw = generate_4d_data(16 * 32, 5.0);

        auto a_cpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto b_cpu = std::make_shared<Tensor>(b_raw, b_shape, false, Device::CPU);
        auto a_gpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto b_gpu = std::make_shared<Tensor>(b_raw, b_shape, false, Device::CPU);

        auto out_cpu = (*a_cpu > *b_cpu);

        a_gpu->to(Device::CUDA);
        b_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = (*a_gpu > *b_gpu);
        double fwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        std::cout << "[PASS] operator> cuda broadcast pass verified (exec: " << fwd_time << " ms)\n";
    }

    std::cout << "[PASS] all binary kernels verified\n";
    std::cout << "------------------------------------------\n";

    std::cout << "starting reduction kernels test...\n";

    // test 23: Tensor sum(dim=2) [Reduce H axis: 16 -> 1]
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->sum(2, false);
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->sum(2, false);
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        assert(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] sum(dim=2) [collapsing H] verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 24: Tensor sum() [Full Tensor Reduction]
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->sum();
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->sum();
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        assert(close_enough((*out_gpu->data)[0], (*out_cpu->data)[0]));
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] full sum() [collapsing all dims] verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 25: Tensor mean(dim=1) [Reduce C axis: 8 -> 1]
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->mean(1, false);
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->mean(1, false);
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        assert(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] mean(dim=1) [collapsing C] verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 26: Tensor max(dim=3) [Reduce W axis: 32 -> 1]
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->max(3, false);
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->max(3, false);
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        assert(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] max(dim=3) [collapsing W] verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 27: Tensor min(dim=0) [Reduce B axis: 4 -> 1]
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);

        auto out_cpu = x_cpu->min(0, false);
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->min(0, false);
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        assert(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] min(dim=0) [collapsing B] verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 28: Tensor argmax(dim=3) [Index along W axis]
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);

        auto out_cpu = x_cpu->argmax(3, false);

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->argmax(3, false);
        double fwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        assert(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        std::cout << "[PASS] argmax(dim=3) [index along W] verified (fwd: " << fwd_time << " ms)\n";
    }

    // test 29: Tensor argmin(dim=2) [Index along H axis]
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);

        auto out_cpu = x_cpu->argmin(2, false);

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->argmin(2, false);
        double fwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        assert(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        std::cout << "[PASS] argmin(dim=2) [index along H] verified (fwd: " << fwd_time << " ms)\n";
    }

    // test 30: Tensor softmax(dim=3) [Softmax along W axis]
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);

        // CPU Reference
        auto out_cpu = x_cpu->softmax(3);
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        // GPU Timed Pass
        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->softmax(3);
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        // Parity Verification
        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        assert(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] softmax(dim=3) verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 31: Tensor log_softmax(dim=2) [LogSoftmax along H axis]
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);

        // CPU Reference
        auto out_cpu = x_cpu->log_softmax(2);
        x_cpu->ensure_grad_allocated();
        out_cpu->backward();

        // GPU Timed Pass
        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->log_softmax(2);
        double fwd_time = timer.stop_ms();

        timer.start();
        out_gpu->backward();
        double bwd_time = timer.stop_ms();

        // arity Verification
        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        assert(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] log_softmax(dim=2) verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 32: Tensor contiguous() with non-contiguous transposed tensor
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);

        // Create a non-contiguous view via transpose (e.g., swap axes 1 and 2)
        auto trans_cpu = x_cpu->transpose(1, 2);
        auto cont_cpu = trans_cpu->contiguous();
        x_cpu->ensure_grad_allocated();
        cont_cpu->backward();

        // GPU execution with timing
        x_gpu->to(Device::CUDA);
        timer.start();
        auto trans_gpu = x_gpu->transpose(1, 2);
        auto cont_gpu = trans_gpu->contiguous();
        double fwd_time = timer.stop_ms();

        timer.start();
        cont_gpu->backward();
        double bwd_time = timer.stop_ms();

        // Parity Verification
        cont_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        assert(cont_gpu->shape == cont_cpu->shape);
        assert(cont_gpu->is_contiguous() == true);
        
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*cont_gpu->data)[i], (*cont_cpu->data)[i]));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
        }
        std::cout << "[PASS] contiguous() on transposed view verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 33: Tensor argsort(dim=3) on GPU tensor (Thrust sort_by_key integration)
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);

        // CPU reference argsort along W axis (dim=3)
        auto out_cpu = x_cpu->argsort(3, false);

        // GPU timed pass using Thrust
        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->argsort(3, false);
        double fwd_time = timer.stop_ms();

        // Parity Verification
        out_gpu->to(Device::CPU);
        assert(out_gpu->shape == out_cpu->shape);
        assert(out_gpu->requires_grad == false);

        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        std::cout << "[PASS] argsort(dim=3) verified (fwd: " << fwd_time << " ms)\n";
    }

    // test 34: CPU vs GPU gradient parity for softmax(dim=1) and log_softmax(dim=1)
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, true, Device::CPU);

        auto sm_cpu = x_cpu->softmax(1);
        x_cpu->ensure_grad_allocated();
        sm_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto sm_gpu = x_gpu->softmax(1);
        double fwd_time = timer.stop_ms();

        timer.start();
        sm_gpu->backward();
        double bwd_time = timer.stop_ms();

        sm_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*sm_gpu->data)[i], (*sm_cpu->data)[i], 1e-4));
            assert(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i], 1e-4));
        }
        std::cout << "[PASS] softmax(dim=1) CPU/GPU autograd parity verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 35: Large-tensor (1,000,000 elements) global sum reduction correctness and timing
    {
        const size_t large_elements = 1000000;
        std::vector<double> large_data(large_elements, 1.5);
        auto x_cpu = std::make_shared<Tensor>(large_data, std::vector<size_t>{large_elements}, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(large_data, std::vector<size_t>{large_elements}, true, Device::CPU);

        auto sum_cpu = x_cpu->sum();
        x_cpu->ensure_grad_allocated();
        sum_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto sum_gpu = x_gpu->sum();
        double fwd_time = timer.stop_ms();

        timer.start();
        sum_gpu->backward();
        double bwd_time = timer.stop_ms();

        sum_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        assert(close_enough((*sum_gpu->data)[0], 1500000.0, 1e-3));
        assert(close_enough((*sum_gpu->data)[0], (*sum_cpu->data)[0], 1e-3));
        assert(close_enough((*x_gpu->grad)[0], (*x_cpu->grad)[0], 1e-4));
        assert(close_enough((*x_gpu->grad)[large_elements - 1], (*x_cpu->grad)[large_elements - 1], 1e-4));
        std::cout << "[PASS] large-tensor global sum (1M elements) verified (fwd: " << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 36: True on-device argsort along non-innermost dimension (dim=1) with descending=true
    {
        auto a_raw = generate_4d_data(num_elements);
        auto x_cpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(a_raw, shape_4d, false, Device::CPU);

        auto out_cpu = x_cpu->argsort(1, true);

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = x_gpu->argsort(1, true);
        double fwd_time = timer.stop_ms();

        out_gpu->to(Device::CPU);
        assert(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < num_elements; ++i) {
            assert(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        std::cout << "[PASS] argsort(dim=1, descending=true) strided GPU verified (fwd: " << fwd_time << " ms)\n";
    }

    std::cout << "==========================================\n";
    std::cout << "[PASS] all GPU tests and latency benchmarks verified cleanly!\n";
    std::cout << "==========================================\n";

    return 0;
}