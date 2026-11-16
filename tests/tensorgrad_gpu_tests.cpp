#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include "test_utils.h"
#include "autograd/Tensor.h"
#include "nn/Conv2D.h"
#include "nn/MaxPool2D.h"
#include "nn/AvgPool2D.h"
#include "nn/Loss.h"
#include "nn/BatchNorm2D.h"
#include "nn/Dropout.h"
#include "nn/Sequential.h"
#include "data/Dataset.h"
#include "data/DataLoader.h"
#include "utils/RNG.h"
#include "optim/SGD.h"
#include "optim/Adam.h"

// helper to assert floating point parity smoothly
bool close_enough(double a, double b, double tol = 1e-4) {
    return std::abs(a - b) < tol;
}

// helper timer to isolate and measure GPU execution latency accurately
struct CudaTimer {
    std::chrono::high_resolution_clock::time_point start_time;

    void start() {
        CHECK_CUDA(cudaDeviceSynchronize());
        start_time = std::chrono::high_resolution_clock::now();
    }

    double stop_ms() {
        CHECK_CUDA(cudaDeviceSynchronize());
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
            CHECK_TENSOR(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
        }
        for (size_t i = 0; i < b_elements; ++i) {
            CHECK_TENSOR(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
            CHECK_TENSOR(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
        }
        for (size_t i = 0; i < b_elements; ++i) {
            CHECK_TENSOR(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
            CHECK_TENSOR(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
        }
        for (size_t i = 0; i < b_elements; ++i) {
            CHECK_TENSOR(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*a_gpu->grad)[i], (*a_cpu->grad)[i]));
            CHECK_TENSOR(close_enough((*b_gpu->grad)[i], (*b_cpu->grad)[i]));
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
            CHECK_TENSOR(close_enough((*a_gpu->data)[i], (*a_cpu->data)[i]));
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
            CHECK_TENSOR(close_enough((*a_gpu->data)[i], (*a_cpu->data)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
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
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
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
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        for (size_t i = 0; i < num_elements; ++i) {
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
        CHECK_TENSOR(close_enough((*out_gpu->data)[0], (*out_cpu->data)[0]));
        for (size_t i = 0; i < num_elements; ++i) {
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        for (size_t i = 0; i < num_elements; ++i) {
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        for (size_t i = 0; i < num_elements; ++i) {
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        for (size_t i = 0; i < num_elements; ++i) {
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
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
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
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
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < num_elements; ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < num_elements; ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
        CHECK_TENSOR(cont_gpu->shape == cont_cpu->shape);
        CHECK_TENSOR(cont_gpu->is_contiguous() == true);
        
        for (size_t i = 0; i < num_elements; ++i) {
            CHECK_TENSOR(close_enough((*cont_gpu->data)[i], (*cont_cpu->data)[i]));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i]));
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
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        CHECK_TENSOR(out_gpu->requires_grad == false);

        for (size_t i = 0; i < num_elements; ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
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
            CHECK_TENSOR(close_enough((*sm_gpu->data)[i], (*sm_cpu->data)[i], 1e-4));
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i], 1e-4));
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
        CHECK_TENSOR(close_enough((*sum_gpu->data)[0], 1500000.0, 1e-3));
        CHECK_TENSOR(close_enough((*sum_gpu->data)[0], (*sum_cpu->data)[0], 1e-3));
        CHECK_TENSOR(close_enough((*x_gpu->grad)[0], (*x_cpu->grad)[0], 1e-4));
        CHECK_TENSOR(close_enough((*x_gpu->grad)[large_elements - 1], (*x_cpu->grad)[large_elements - 1], 1e-4));
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
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < num_elements; ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i]));
        }
        std::cout << "[PASS] argsort(dim=1, descending=true) strided GPU verified (fwd: " << fwd_time << " ms)\n";
    }

    // test 37: Conv2D forward (im2col) and backward (col2im) CPU vs GPU parity
    {
        // Batch=2, InChannels=3, Height=8, Width=8
        const size_t B = 2, C_in = 3, H = 8, W = 8;
        const size_t C_out = 4, K = 3, stride = 1, pad = 1;
        const size_t total_in = B * C_in * H * W;

        auto in_raw = generate_4d_data(total_in, 1.0, 0.5);
        auto x_cpu = std::make_shared<Tensor>(in_raw, std::vector<size_t>{B, C_in, H, W}, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(in_raw, std::vector<size_t>{B, C_in, H, W}, true, Device::CPU);

        auto conv_cpu = std::make_shared<Conv2D>(C_in, C_out, K, stride, pad);
        auto conv_gpu = std::make_shared<Conv2D>(C_in, C_out, K, stride, pad);

        // Synchronize initial weights and bias so both layers start identical
        conv_gpu->weight = std::make_shared<Tensor>(*conv_cpu->weight->data, conv_cpu->weight->shape, true, Device::CPU);
        conv_gpu->bias = std::make_shared<Tensor>(*conv_cpu->bias->data, conv_cpu->bias->shape, true, Device::CPU);

        // 1. CPU forward & backward pass
        auto out_cpu = conv_cpu->forward(x_cpu);
        auto loss_cpu = out_cpu->sum();
        x_cpu->ensure_grad_allocated();
        loss_cpu->backward();

        // 2. GPU forward & backward pass with latency timing
        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = conv_gpu->forward(x_gpu);
        double fwd_time = timer.stop_ms();

        auto loss_gpu = out_gpu->sum();
        timer.start();
        loss_gpu->backward();
        double bwd_time = timer.stop_ms();

        // 3. Move GPU results back to host for assertion
        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        conv_gpu->weight->to(Device::CPU);
        conv_gpu->bias->to(Device::CPU);

        // Verify forward output activations
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i], 1e-3));
        }

        // Verify col2im input image gradient routing
        for (size_t i = 0; i < total_in; ++i) {
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i], 1e-3));
        }

        // Verify filter weight gradients
        for (size_t i = 0; i < conv_cpu->weight->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*conv_gpu->weight->grad)[i], (*conv_cpu->weight->grad)[i], 1e-3));
        }

        // Verify channel bias gradients
        for (size_t i = 0; i < conv_cpu->bias->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*conv_gpu->bias->grad)[i], (*conv_cpu->bias->grad)[i], 1e-3));
        }

        std::cout << "[PASS] Conv2D forward (im2col) & backward (col2im) CPU/GPU parity verified (fwd: " 
                  << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 38: MaxPool2D forward and backward CPU vs GPU parity
    {
        // Batch=2, Channels=3, Height=8, Width=8, Kernel=2, Stride=2
        const size_t B = 2, C = 3, H = 8, W = 8;
        const size_t K = 2, stride = 2;
        const size_t total_in = B * C * H * W;

        auto in_raw = generate_4d_data(total_in, 1.0, 0.5);
        auto x_cpu = std::make_shared<Tensor>(in_raw, std::vector<size_t>{B, C, H, W}, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(in_raw, std::vector<size_t>{B, C, H, W}, true, Device::CPU);

        auto pool_cpu = std::make_shared<MaxPool2D>(K, stride);
        auto pool_gpu = std::make_shared<MaxPool2D>(K, stride);

        // CPU pass
        auto out_cpu = pool_cpu->forward(x_cpu);
        auto loss_cpu = out_cpu->sum();
        x_cpu->ensure_grad_allocated();
        loss_cpu->backward();

        // GPU pass
        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = pool_gpu->forward(x_gpu);
        double fwd_time = timer.stop_ms();

        auto loss_gpu = out_gpu->sum();
        timer.start();
        loss_gpu->backward();
        double bwd_time = timer.stop_ms();

        // Transfer back for verification
        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);

        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i], 1e-4));
        }

        for (size_t i = 0; i < total_in; ++i) {
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i], 1e-4));
        }

        std::cout << "[PASS] MaxPool2D forward & backward CPU/GPU autograd parity verified (fwd: "
                  << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // test 39: AvgPool2D forward and backward CPU vs GPU parity
    {
        // Batch=2, Channels=3, Height=8, Width=8, Kernel=2, Stride=2
        const size_t B = 2, C = 3, H = 8, W = 8;
        const size_t K = 2, stride = 2;
        const size_t total_in = B * C * H * W;

        auto in_raw = generate_4d_data(total_in, 1.0, 0.5);
        auto x_cpu = std::make_shared<Tensor>(in_raw, std::vector<size_t>{B, C, H, W}, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(in_raw, std::vector<size_t>{B, C, H, W}, true, Device::CPU);

        auto pool_cpu = std::make_shared<AvgPool2D>(K, stride);
        auto pool_gpu = std::make_shared<AvgPool2D>(K, stride);

        // 1. CPU forward & backward pass
        auto out_cpu = pool_cpu->forward(x_cpu);
        auto loss_cpu = out_cpu->sum();
        x_cpu->ensure_grad_allocated();
        loss_cpu->backward();

        // 2. GPU forward & backward pass with timing
        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = pool_gpu->forward(x_gpu);
        double fwd_time = timer.stop_ms();

        auto loss_gpu = out_gpu->sum();
        timer.start();
        loss_gpu->backward();
        double bwd_time = timer.stop_ms();

        // 3. Transfer back for parity check
        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);

        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < out_cpu->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i], 1e-4));
        }

        for (size_t i = 0; i < total_in; ++i) {
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i], 1e-4));
        }

        std::cout << "[PASS] AvgPool2D forward & backward CPU/GPU autograd parity verified (fwd: "
                  << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

        // test 40: MaxPool2D and AvgPool2D rank and underflow bounds guards
    {
        auto max_pool = std::make_shared<MaxPool2D>(3, 1);
        auto avg_pool = std::make_shared<AvgPool2D>(3, 1);

        // 1. Test invalid rank (2D tensor instead of 4D)
        auto invalid_2d = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0}, std::vector<size_t>{2, 2}, false, Device::CPU);
        bool caught_max_rank = false;
        bool caught_avg_rank = false;

        try {
            max_pool->forward(invalid_2d);
        } catch (const std::invalid_argument&) {
            caught_max_rank = true;
        }

        try {
            avg_pool->forward(invalid_2d);
        } catch (const std::invalid_argument&) {
            caught_avg_rank = true;
        }

        CHECK_TENSOR(caught_max_rank);
        CHECK_TENSOR(caught_avg_rank);

        // 2. Test underflow hazard (image 2x2 with kernel 3x3)
        auto small_img = std::make_shared<Tensor>(std::vector<double>(4, 1.0), std::vector<size_t>{1, 1, 2, 2}, false, Device::CUDA);
        bool caught_max_underflow = false;
        bool caught_avg_underflow = false;

        try {
            max_pool->forward(small_img);
        } catch (const std::invalid_argument&) {
            caught_max_underflow = true;
        }

        try {
            avg_pool->forward(small_img);
        } catch (const std::invalid_argument&) {
            caught_avg_underflow = true;
        }

        CHECK_TENSOR(caught_max_underflow);
        CHECK_TENSOR(caught_avg_underflow);

        std::cout << "[PASS] MaxPool2D & AvgPool2D 4D rank guards & underflow checks verified\n";
    }

        // test 41: GPU-native SparseCategoricalCrossEntropy forward and backward parity
    {
        // Batch=2, Classes=3
        std::vector<double> scce_logits = {
            1.0, 2.0, 3.0,   // sample 0 logits
            1.0, 5.0, 1.0    // sample 1 logits
        };
        std::vector<double> scce_targets = {2.0, 1.0}; // sample 0 -> class 2, sample 1 -> class 1

        auto logits_cpu = std::make_shared<Tensor>(scce_logits, std::vector<size_t>{2, 3}, true, Device::CPU);
        auto targets_cpu = std::make_shared<Tensor>(scce_targets, std::vector<size_t>{2}, false, Device::CPU);

        auto logits_gpu = std::make_shared<Tensor>(scce_logits, std::vector<size_t>{2, 3}, true, Device::CUDA);
        auto targets_gpu = std::make_shared<Tensor>(scce_targets, std::vector<size_t>{2}, false, Device::CUDA);

        SparseCategoricalCrossEntropyLoss criterion;

        // CPU pass
        auto loss_cpu = criterion(logits_cpu, targets_cpu);
        logits_cpu->ensure_grad_allocated();
        loss_cpu->backward();

        // GPU pass with timing
        timer.start();
        auto loss_gpu = criterion(logits_gpu, targets_gpu);
        double fwd_time = timer.stop_ms();

        logits_gpu->ensure_grad_allocated();
        timer.start();
        loss_gpu->backward();
        double bwd_time = timer.stop_ms();

        // Transfer GPU results back for parity check
        loss_gpu->to(Device::CPU);
        logits_gpu->to(Device::CPU);

        // Analytical baseline check: loss ~ 0.221791
        CHECK_TENSOR(close_enough((*loss_gpu->data)[0], (*loss_cpu->data)[0], 1e-4));
        CHECK_TENSOR(close_enough((*loss_gpu->data)[0], 0.22179113, 1e-4));

        // Gradients check on logits:
        // sample 0, class 2: (prob - 1) / N = -0.16737953
        CHECK_TENSOR(close_enough((*logits_gpu->grad)[2], (*logits_cpu->grad)[2], 1e-4));
        // sample 1, class 1: (prob - 1) / N = -0.01766872
        CHECK_TENSOR(close_enough((*logits_gpu->grad)[4], (*logits_cpu->grad)[4], 1e-4));

        // Full gradient tensor parity across all classes
        for (size_t i = 0; i < logits_cpu->data->size(); ++i) {
            CHECK_TENSOR(close_enough((*logits_gpu->grad)[i], (*logits_cpu->grad)[i], 1e-4));
        }

        std::cout << "[PASS] GPU-native SparseCategoricalCrossEntropy forward & backward verified (fwd: "
                  << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

        // test 42: SparseCategoricalCrossEntropy negative target clamping and bounds guard
    {
        SparseCategoricalCrossEntropyLoss criterion;

        // 1. CPU validation: Negative target must throw std::out_of_range
        auto logits_cpu = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0}, std::vector<size_t>{1, 3}, true, Device::CPU);
        auto neg_target_cpu = std::make_shared<Tensor>(std::vector<double>{-1.0}, std::vector<size_t>{1}, false, Device::CPU);
        bool caught_cpu_neg = false;
        try {
            criterion(logits_cpu, neg_target_cpu);
        } catch (const std::out_of_range&) {
            caught_cpu_neg = true;
        }
        CHECK_TENSOR(caught_cpu_neg);

        // 2. GPU validation: Ensure negative target (-1.0) does NOT silently alias to class 0
        // Batch=2: Sample 0 has target -1.0 (invalid), Sample 1 has target 2.0 (class 2)
        std::vector<double> sample_logits = {
            2.0, 1.0, 0.5,   // Sample 0: class 0 logit is 2.0
            0.5, 1.0, 2.0    // Sample 1: class 2 logit is 2.0
        };
        auto logits_gpu = std::make_shared<Tensor>(sample_logits, std::vector<size_t>{2, 3}, true, Device::CUDA);
        auto targets_gpu = std::make_shared<Tensor>(std::vector<double>{-1.0, 2.0}, std::vector<size_t>{2}, false, Device::CUDA);

        logits_gpu->ensure_grad_allocated();
        auto loss_gpu = criterion(logits_gpu, targets_gpu);
        loss_gpu->backward();

        logits_gpu->to(Device::CPU);

        // If -1.0 aliased to class 0, Sample 0 class 0 gradient (logits_gpu->grad[0]) would be negative.
        // With the guard, Sample 0 is ignored, so Sample 0 class 0 gradient must remain 0.0!
        CHECK_TENSOR(close_enough((*logits_gpu->grad)[0], 0.0, 1e-5));

        // Sample 1 target is class 2, so it should receive normal gradient:
        // (prob - 1) / N != 0.0
        CHECK_TENSOR((*logits_gpu->grad)[5] < 0.0);

        std::cout << "[PASS] SparseCategoricalCrossEntropy PTX negative clamping guard verified\n";
    }

        // test 43: BatchNorm2D forward, backward, running stats, and evaluation mode on GPU
    {
        // Batch=2, Channels=3, Height=4, Width=4
        const size_t B = 2, C = 3, H = 4, W = 4;
        const size_t total_elements = B * C * H * W;

        auto raw_in = generate_4d_data(total_elements, 2.0, 1.5);
        auto x_cpu = std::make_shared<Tensor>(raw_in, std::vector<size_t>{B, C, H, W}, true, Device::CPU);
        auto x_gpu = std::make_shared<Tensor>(raw_in, std::vector<size_t>{B, C, H, W}, true, Device::CPU);

        auto bn_cpu = std::make_shared<BatchNorm2D>(C);
        auto bn_gpu = std::make_shared<BatchNorm2D>(C);

        // 1. Training mode forward & backward pass
        auto out_cpu = bn_cpu->forward(x_cpu);
        auto loss_cpu = out_cpu->sum();
        x_cpu->ensure_grad_allocated();
        loss_cpu->backward();

        x_gpu->to(Device::CUDA);
        timer.start();
        auto out_gpu = bn_gpu->forward(x_gpu);
        double fwd_time = timer.stop_ms();

        auto loss_gpu = out_gpu->sum();
        timer.start();
        loss_gpu->backward();
        double bwd_time = timer.stop_ms();

        // Transfer GPU results back for parity check
        out_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);
        bn_gpu->weight->to(Device::CPU);
        bn_gpu->bias->to(Device::CPU);
        bn_gpu->running_mean->to(Device::CPU);
        bn_gpu->running_var->to(Device::CPU);

        // Verify activations parity
        CHECK_TENSOR(out_gpu->shape == out_cpu->shape);
        for (size_t i = 0; i < total_elements; ++i) {
            CHECK_TENSOR(close_enough((*out_gpu->data)[i], (*out_cpu->data)[i], 1e-4));
        }

        // Verify gradient routing parity
        for (size_t i = 0; i < total_elements; ++i) {
            CHECK_TENSOR(close_enough((*x_gpu->grad)[i], (*x_cpu->grad)[i], 1e-4));
        }

        // Verify weight and bias gradient parity
        for (size_t c = 0; c < C; ++c) {
            CHECK_TENSOR(close_enough((*bn_gpu->weight->grad)[c], (*bn_cpu->weight->grad)[c], 1e-4));
            CHECK_TENSOR(close_enough((*bn_gpu->bias->grad)[c], (*bn_cpu->bias->grad)[c], 1e-4));
        }

        // Verify running statistics EMA updates parity
        for (size_t c = 0; c < C; ++c) {
            CHECK_TENSOR(close_enough((*bn_gpu->running_mean->data)[c], (*bn_cpu->running_mean->data)[c], 1e-4));
            CHECK_TENSOR(close_enough((*bn_gpu->running_var->data)[c], (*bn_cpu->running_var->data)[c], 1e-4));
        }

        // 2. Evaluation / Inference mode test
        bn_cpu->training = false;
        bn_gpu->training = false;

        auto eval_x_cpu = std::make_shared<Tensor>(raw_in, std::vector<size_t>{B, C, H, W}, false, Device::CPU);
        auto eval_x_gpu = std::make_shared<Tensor>(raw_in, std::vector<size_t>{B, C, H, W}, false, Device::CUDA);

        auto eval_out_cpu = bn_cpu->forward(eval_x_cpu);
        auto eval_out_gpu = bn_gpu->forward(eval_x_gpu);
        eval_out_gpu->to(Device::CPU);

        for (size_t i = 0; i < total_elements; ++i) {
            CHECK_TENSOR(close_enough((*eval_out_gpu->data)[i], (*eval_out_cpu->data)[i], 1e-4));
        }

        std::cout << "[PASS] BatchNorm2D CUDA training, autograd, EMA stats, and eval inference verified (fwd: "
                  << fwd_time << " ms, bwd: " << bwd_time << " ms)\n";
    }

    // -------------------------------------------------------------
    // TEST 44: GPU-Native SGD Optimizer Parity and Momentum Tracking
    // -------------------------------------------------------------
    {
        const size_t N = 1024;
        std::vector<double> init_w(N);
        std::vector<double> init_g(N);
        for (size_t i = 0; i < N; ++i) {
            init_w[i] = std::sin(static_cast<double>(i) * 0.1);
            init_g[i] = std::cos(static_cast<double>(i) * 0.1);
        }

        // CPU parameter setup
        auto w_cpu = std::make_shared<Tensor>(init_w, std::vector<size_t>{N}, true, Device::CPU);
        w_cpu->ensure_grad_allocated();
        std::copy(init_g.begin(), init_g.end(), w_cpu->grad->begin());

        // GPU parameter setup
        auto w_gpu = std::make_shared<Tensor>(init_w, std::vector<size_t>{N}, true, Device::CUDA);
        w_gpu->ensure_grad_allocated();
        CUDA_CHECK(cudaMemcpy(w_gpu->cuda_grad.get(), init_g.data(), N * sizeof(double), cudaMemcpyHostToDevice));

        // Optimizers with momentum and weight decay
        const double lr = 0.05;
        const double momentum = 0.9;
        const double wd = 0.01;

        SGD opt_cpu({w_cpu}, lr, momentum, wd);
        SGD opt_gpu({w_gpu}, lr, momentum, wd);

        // Benchmark multi-step execution parity
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int step = 0; step < 5; ++step) {
            opt_gpu.step();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float gpu_time = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // Execute corresponding steps on CPU
        for (int step = 0; step < 5; ++step) {
            opt_cpu.step();
        }

        // Copy back GPU weights to verify analytical numerical parity
        w_gpu->to(Device::CPU);
        for (size_t i = 0; i < N; ++i) {
            CHECK_TENSOR(close_enough((*w_gpu->data)[i], (*w_cpu->data)[i], 1e-5));
        }

        std::cout << "[PASS] GPU-native SGD momentum & weight decay step verified (5 steps: "
                  << gpu_time << " ms)\n";
    }

    // -------------------------------------------------------------
    // TEST 45: GPU-Native Adam Optimizer Parity (Standard & AMSGrad)
    // -------------------------------------------------------------
    {
        const size_t N = 1024;
        std::vector<double> init_w(N);
        std::vector<double> init_g(N);
        for (size_t i = 0; i < N; ++i) {
            init_w[i] = std::sin(static_cast<double>(i) * 0.05);
            init_g[i] = std::cos(static_cast<double>(i) * 0.05);
        }

        // 1. Standard Adam Test
        {
            auto w_cpu = std::make_shared<Tensor>(init_w, std::vector<size_t>{N}, true, Device::CPU);
            w_cpu->ensure_grad_allocated();
            std::copy(init_g.begin(), init_g.end(), w_cpu->grad->begin());

            auto w_gpu = std::make_shared<Tensor>(init_w, std::vector<size_t>{N}, true, Device::CUDA);
            w_gpu->ensure_grad_allocated();
            CUDA_CHECK(cudaMemcpy(w_gpu->cuda_grad.get(), init_g.data(), N * sizeof(double), cudaMemcpyHostToDevice));

            const double lr = 0.01;
            const double wd = 0.01;

            Adam opt_cpu({w_cpu}, lr, {0.9, 0.999}, 1e-8, wd, false, false);
            Adam opt_gpu({w_gpu}, lr, {0.9, 0.999}, 1e-8, wd, false, false);

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start));
            for (int step = 0; step < 5; ++step) {
                opt_gpu.step();
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float gpu_time = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            for (int step = 0; step < 5; ++step) {
                opt_cpu.step();
            }

            w_gpu->to(Device::CPU);
            for (size_t i = 0; i < N; ++i) {
                CHECK_TENSOR(close_enough((*w_gpu->data)[i], (*w_cpu->data)[i], 1e-5));
            }

            std::cout << "[PASS] GPU-native Adam standard step verified (5 steps: "
                      << gpu_time << " ms)\n";
        }

        // 2. AMSGrad Variant Test
        {
            auto w_cpu_ams = std::make_shared<Tensor>(init_w, std::vector<size_t>{N}, true, Device::CPU);
            w_cpu_ams->ensure_grad_allocated();
            std::copy(init_g.begin(), init_g.end(), w_cpu_ams->grad->begin());

            auto w_gpu_ams = std::make_shared<Tensor>(init_w, std::vector<size_t>{N}, true, Device::CUDA);
            w_gpu_ams->ensure_grad_allocated();
            CUDA_CHECK(cudaMemcpy(w_gpu_ams->cuda_grad.get(), init_g.data(), N * sizeof(double), cudaMemcpyHostToDevice));

            const double lr = 0.01;
            const double wd = 0.01;

            Adam opt_cpu_ams({w_cpu_ams}, lr, {0.9, 0.999}, 1e-8, wd, true, false);
            Adam opt_gpu_ams({w_gpu_ams}, lr, {0.9, 0.999}, 1e-8, wd, true, false);

            for (int step = 0; step < 5; ++step) {
                opt_gpu_ams.step();
                opt_cpu_ams.step();
            }

            w_gpu_ams->to(Device::CPU);
            for (size_t i = 0; i < N; ++i) {
                CHECK_TENSOR(close_enough((*w_gpu_ams->data)[i], (*w_cpu_ams->data)[i], 1e-5));
            }

            std::cout << "[PASS] GPU-native Adam AMSGrad variant step verified\n";
        }
    }

    // -------------------------------------------------------------
    // TEST 46: GPU-Native Inverted Dropout (Expectation, Grad Routing, Eval & Parity)
    // -------------------------------------------------------------
    {
        // 1. Constructor range validation: p must lie in [0.0, 1.0)
        bool caught_neg = false, caught_one = false, caught_gt_one = false;
        try { Dropout d(-0.1); } catch (const std::invalid_argument&) { caught_neg = true; }
        try { Dropout d(1.0); }  catch (const std::invalid_argument&) { caught_one = true; }
        try { Dropout d(1.5); }  catch (const std::invalid_argument&) { caught_gt_one = true; }
        CHECK_TENSOR(caught_neg && caught_one && caught_gt_one);

        // 2. Statistical expectation E[out] ~= x, inverted scaling, and exact grad routing on GPU
        RNG::manual_seed(42);
        const size_t N = 100000;
        const double p = 0.4;
        const double scale = 1.0 / (1.0 - p);
        const double const_val = 3.0;

        auto x_gpu = std::make_shared<Tensor>(
            std::vector<double>(N, const_val), std::vector<size_t>{100, 1000}, true, Device::CUDA
        );
        // Pre-seed x_gpu->grad with 1.0 everywhere to verify backward accumulates (+=) for residual paths
        x_gpu->ensure_grad_allocated();
        std::vector<double> ones_grad(N, 1.0);
        CHECK_CUDA(cudaMemcpy(x_gpu->cuda_grad.get(), ones_grad.data(), N * sizeof(double), cudaMemcpyHostToDevice));

        auto drop = std::make_shared<Dropout>(p);
        timer.start();
        auto y_gpu = drop->forward(x_gpu);
        double fwd_ms = timer.stop_ms();

        auto loss_gpu = y_gpu->sum();
        timer.start();
        loss_gpu->backward();
        double bwd_ms = timer.stop_ms();

        y_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);

        double sum_out = 0.0;
        size_t zero_count = 0;
        for (size_t i = 0; i < N; ++i) {
            double yi = (*y_gpu->data)[i];
            double gi = (*x_gpu->grad)[i];
            if (yi == 0.0) {
                ++zero_count;
                // Dropped unit: upstream grad must be masked to 0, leaving pre-existing grad (1.0) untouched
                CHECK_CLOSE(gi, 1.0, 1e-12);
            } else {
                // Kept unit: output scaled by 1/(1-p), grad accumulated by +1/(1-p)
                CHECK_CLOSE(yi, const_val * scale, 1e-12);
                CHECK_CLOSE(gi, 1.0 + scale, 1e-12);
            }
            sum_out += yi;
        }

        double empirical_drop_rate = static_cast<double>(zero_count) / static_cast<double>(N);
        double empirical_mean = sum_out / static_cast<double>(N);
        CHECK_CLOSE(empirical_drop_rate, p, 0.015);
        CHECK_CLOSE(empirical_mean, const_val, 0.05);

        // 3. Deterministic CPU vs CUDA bitwise parity under RNG::manual_seed
        const size_t M = 2048;
        std::vector<double> vals(M);
        for (size_t i = 0; i < M; ++i) vals[i] = static_cast<double>(i % 17) * 0.25 - 2.0;

        Dropout drop_parity(0.35);

        RNG::manual_seed(2026);
        auto x_cpu_p = std::make_shared<Tensor>(vals, std::vector<size_t>{32, 64}, true, Device::CPU);
        auto y_cpu_p = drop_parity.forward(x_cpu_p);
        y_cpu_p->sum()->backward();

        RNG::manual_seed(2026);
        auto x_gpu_p = std::make_shared<Tensor>(vals, std::vector<size_t>{32, 64}, true, Device::CUDA);
        auto y_gpu_p = drop_parity.forward(x_gpu_p);
        y_gpu_p->sum()->backward();

        y_gpu_p->to(Device::CPU);
        x_gpu_p->to(Device::CPU);
        for (size_t i = 0; i < M; ++i) {
            CHECK_CLOSE((*y_gpu_p->data)[i], (*y_cpu_p->data)[i], 1e-12);
            CHECK_CLOSE((*x_gpu_p->grad)[i], (*x_cpu_p->grad)[i], 1e-12);
        }

        // 4. eval() toggle (direct & via Sequential) and p = 0.0 identity pass-through
        Sequential seq;
        seq.add(drop);
        seq.eval();
        CHECK_TENSOR(drop->training == false);

        auto eval_in = std::make_shared<Tensor>(vals, std::vector<size_t>{32, 64}, true, Device::CUDA);
        auto eval_out = seq.forward(eval_in);
        CHECK_TENSOR(eval_out.get() == eval_in.get()); // exact zero-copy identity pass-through

        seq.train();
        CHECK_TENSOR(drop->training == true);

        Dropout zero_drop(0.0);
        auto zero_out = zero_drop.forward(eval_in);
        CHECK_TENSOR(zero_out.get() == eval_in.get());

        std::cout << "[PASS] GPU-native Dropout expectation, grad routing, eval toggle & CPU/GPU parity verified (fwd: "
                  << fwd_ms << " ms, bwd: " << bwd_ms << " ms)\n";
    }

    // test 47: dropout2d spatial channel masking, grad routing, eval, and cpu/gpu parity
    {
        auto expect_invalid_arg = [](auto&& fn) {
            try {
                fn();
                return false;
            } catch (const std::invalid_argument&) {
                return true;
            }
        };

        CHECK_TENSOR(expect_invalid_arg([]() { Dropout2D(-0.1); }));
        CHECK_TENSOR(expect_invalid_arg([]() { Dropout2D(1.0); }));

        auto d2_guard = std::make_shared<Dropout2D>(0.5);
        auto t2d = std::make_shared<Tensor>(std::vector<double>(6, 1.0), std::vector<size_t>{2, 3}, false, Device::CUDA);
        auto t3d = std::make_shared<Tensor>(std::vector<double>(12, 1.0), std::vector<size_t>{2, 2, 3}, false, Device::CUDA);
        CHECK_TENSOR(expect_invalid_arg([&]() { d2_guard->forward(t2d); }));

        // rank validation must still fire when eval mode bypasses masking
        d2_guard->eval();
        CHECK_TENSOR(expect_invalid_arg([&]() { d2_guard->forward(t3d); }));

        constexpr size_t B = 64, C = 128, H = 4, W = 4;
        constexpr size_t spatial_size = H * W;
        constexpr size_t num_planes = B * C;
        constexpr size_t total_elements = num_planes * spatial_size;
        constexpr double drop_prob = 0.35;
        constexpr double inv_keep_scale = 1.0 / (1.0 - drop_prob);
        constexpr double pre_grad = 1.0;
        constexpr double strict_tol = 1e-12;
        constexpr double drop_rate_tol = 0.025;
        constexpr double mean_tol = 0.12;
        constexpr unsigned int test_seed = 1337;

        // positive offset keeps every entry strictly > 0 so zeroed planes are unambiguous
        auto x_vals = generate_4d_data(total_elements, 20.0, 0.25);
        auto w_vals = generate_4d_data(total_elements, 15.0, 0.10);
        double expected_in_mean = 0.0;
        for (double v : x_vals) expected_in_mean += v;
        expected_in_mean /= static_cast<double>(total_elements);

        auto x_gpu = std::make_shared<Tensor>(x_vals, std::vector<size_t>{B, C, H, W}, true, Device::CUDA);
        auto w_gpu = std::make_shared<Tensor>(w_vals, std::vector<size_t>{B, C, H, W}, false, Device::CUDA);

        // pre-seed grad to catch accidental overwrites instead of += accumulation
        x_gpu->ensure_grad_allocated();
        std::vector<double> seed_grad(total_elements, pre_grad);
        CHECK_CUDA(cudaMemcpy(x_gpu->cuda_grad.get(), seed_grad.data(), total_elements * sizeof(double), cudaMemcpyHostToDevice));

        RNG::manual_seed(test_seed);
        auto drop2d = std::make_shared<Dropout2D>(drop_prob);
        timer.start();
        auto y_gpu = drop2d->forward(x_gpu);
        double fwd_ms = timer.stop_ms();

        auto weighted_loss = (y_gpu * w_gpu)->sum();
        timer.start();
        weighted_loss->backward();
        double bwd_ms = timer.stop_ms();

        y_gpu->to(Device::CPU);
        x_gpu->to(Device::CPU);

        size_t dropped_planes = 0;
        double out_sum = 0.0;
        for (size_t bc = 0; bc < num_planes; ++bc) {
            size_t base = bc * spatial_size;
            bool plane_kept = ((*y_gpu->data)[base] != 0.0);
            if (!plane_kept) ++dropped_planes;

            for (size_t s = 0; s < spatial_size; ++s) {
                size_t idx = base + s;
                double yi = (*y_gpu->data)[idx];
                double gi = (*x_gpu->grad)[idx];
                if (!plane_kept) {
                    CHECK_CLOSE(yi, 0.0, strict_tol);
                    CHECK_CLOSE(gi, pre_grad, strict_tol);
                } else {
                    CHECK_CLOSE(yi, x_vals[idx] * inv_keep_scale, strict_tol);
                    CHECK_CLOSE(gi, pre_grad + w_vals[idx] * inv_keep_scale, strict_tol);
                }
                out_sum += yi;
            }
        }

        double empirical_drop_rate = static_cast<double>(dropped_planes) / static_cast<double>(num_planes);
        double empirical_out_mean = out_sum / static_cast<double>(total_elements);
        CHECK_CLOSE(empirical_drop_rate, drop_prob, drop_rate_tol);
        CHECK_CLOSE(empirical_out_mean, expected_in_mean, mean_tol);

        // cpu openmp vs cuda parity under matching seed
        constexpr size_t pB = 4, pC = 8, pH = 5, pW = 5;
        constexpr size_t pTotal = pB * pC * pH * pW;
        constexpr unsigned int parity_seed = 9001;
        auto p_vals = generate_4d_data(pTotal, -2.0, 0.5);
        auto pw_vals = generate_4d_data(pTotal, 12.0, 0.1);

        Dropout2D drop2d_parity(0.4);

        RNG::manual_seed(parity_seed);
        auto x_cpu_p = std::make_shared<Tensor>(p_vals, std::vector<size_t>{pB, pC, pH, pW}, true, Device::CPU);
        auto w_cpu_p = std::make_shared<Tensor>(pw_vals, std::vector<size_t>{pB, pC, pH, pW}, false, Device::CPU);
        auto y_cpu_p = drop2d_parity.forward(x_cpu_p);
        (y_cpu_p * w_cpu_p)->sum()->backward();

        RNG::manual_seed(parity_seed);
        auto x_gpu_p = std::make_shared<Tensor>(p_vals, std::vector<size_t>{pB, pC, pH, pW}, true, Device::CUDA);
        auto w_gpu_p = std::make_shared<Tensor>(pw_vals, std::vector<size_t>{pB, pC, pH, pW}, false, Device::CUDA);
        auto y_gpu_p = drop2d_parity.forward(x_gpu_p);
        (y_gpu_p * w_gpu_p)->sum()->backward();

        y_gpu_p->to(Device::CPU);
        x_gpu_p->to(Device::CPU);
        for (size_t i = 0; i < pTotal; ++i) {
            CHECK_CLOSE((*y_gpu_p->data)[i], (*y_cpu_p->data)[i], strict_tol);
            CHECK_CLOSE((*x_gpu_p->grad)[i], (*x_cpu_p->grad)[i], strict_tol);
        }

        // eval bypass and p = 0 must return the exact input pointer without allocating
        Sequential seq;
        seq.add(drop2d);
        seq.eval();
        CHECK_TENSOR(drop2d->training == false);
        CHECK_TENSOR(seq.forward(x_gpu_p).get() == x_gpu_p.get());

        seq.train();
        CHECK_TENSOR(drop2d->training == true);

        Dropout2D zero_drop2d(0.0);
        CHECK_TENSOR(zero_drop2d.forward(x_gpu_p).get() == x_gpu_p.get());

        std::cout << "[PASS] GPU-native Dropout2D channel-plane masking, grad routing, eval toggle & CPU/GPU parity verified (fwd: "
                  << fwd_ms << " ms, bwd: " << bwd_ms << " ms)\n";
    }

    // test 48: dataset/dataloader shuffling, batch collation, cuda transfer, and multi-epoch leak check
    {
        auto expect_invalid_arg = [](auto&& fn) {
            try {
                fn();
                return false;
            } catch (const std::invalid_argument&) {
                return true;
            }
        };

        auto expect_out_of_range = [](auto&& fn) {
            try {
                fn();
                return false;
            } catch (const std::out_of_range&) {
                return true;
            }
        };

        constexpr size_t N = 10;
        constexpr size_t C = 2, H = 3, W = 4;
        constexpr size_t sample_elems = C * H * W;
        constexpr size_t batch_size = 4;
        constexpr double sample_tag_scale = 100.0;
        constexpr double strict_tol = 1e-12;

        std::vector<double> x_vals(N * sample_elems);
        std::vector<double> y_vals(N);
        for (size_t i = 0; i < N; ++i) {
            y_vals[i] = static_cast<double>(i);
            for (size_t k = 0; k < sample_elems; ++k) {
                x_vals[i * sample_elems + k] = static_cast<double>(i) * sample_tag_scale + static_cast<double>(k);
            }
        }

        // construct on cuda to verify dataset normalizes to host without mutating caller tensor
        auto x_cuda = std::make_shared<Tensor>(x_vals, std::vector<size_t>{N, C, H, W}, false, Device::CUDA);
        auto y_cpu = std::make_shared<Tensor>(y_vals, std::vector<size_t>{N}, false, Device::CPU);
        auto bad_y = std::make_shared<Tensor>(std::vector<double>(N + 1, 0.0), std::vector<size_t>{N + 1}, false, Device::CPU);

        CHECK_TENSOR(expect_invalid_arg([&]() { TensorDataset(nullptr, y_cpu); }));
        CHECK_TENSOR(expect_invalid_arg([&]() { TensorDataset(x_cuda, bad_y); }));

        auto ds = std::make_shared<TensorDataset>(x_cuda, y_cpu);
        CHECK_EQUAL(ds->size(), N);
        CHECK_TENSOR(x_cuda->device == Device::CUDA);
        CHECK_TENSOR(expect_out_of_range([&]() { ds->get(N); }));

        auto [s3_x, s3_y] = ds->get(3);
        CHECK_TENSOR(s3_x->shape == (std::vector<size_t>{C, H, W}));
        CHECK_TENSOR(s3_y->shape.empty());
        CHECK_CLOSE((*s3_y->data)[0], 3.0, strict_tol);
        for (size_t k = 0; k < sample_elems; ++k) {
            CHECK_CLOSE((*s3_x->data)[k], 3.0 * sample_tag_scale + static_cast<double>(k), strict_tol);
        }

        CHECK_TENSOR(expect_invalid_arg([&]() { DataLoader(nullptr, batch_size); }));
        CHECK_TENSOR(expect_invalid_arg([&]() { DataLoader(ds, 0); }));

        // drop_last=false keeps partial final batch; drop_last=true truncates it
        DataLoader seq_loader(ds, batch_size, false, false, Device::CPU);
        CHECK_EQUAL(seq_loader.num_batches(), size_t{3});
        std::vector<size_t> seq_sizes;
        std::vector<size_t> seq_targets;
        for (auto& [bx, by] : seq_loader) {
            CHECK_TENSOR(bx->device == Device::CPU);
            CHECK_TENSOR(by->device == Device::CPU);
            seq_sizes.push_back(bx->shape[0]);
            CHECK_TENSOR(bx->shape == (std::vector<size_t>{bx->shape[0], C, H, W}));
            CHECK_TENSOR(by->shape == (std::vector<size_t>{bx->shape[0]}));
            for (size_t b = 0; b < bx->shape[0]; ++b) {
                seq_targets.push_back(static_cast<size_t>((*by->data)[b]));
            }
        }
        CHECK_TENSOR(seq_sizes == (std::vector<size_t>{4, 4, 2}));
        CHECK_TENSOR(seq_targets == (std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

        DataLoader drop_loader(ds, batch_size, false, true, Device::CPU);
        CHECK_EQUAL(drop_loader.num_batches(), size_t{2});
        std::vector<size_t> drop_sizes;
        for (auto& [bx, by] : drop_loader) {
            drop_sizes.push_back(bx->shape[0]);
            CHECK_EQUAL(by->shape[0], bx->shape[0]);
        }
        CHECK_TENSOR(drop_sizes == (std::vector<size_t>{4, 4}));

        // shuffle=true with cuda collation must keep (x, y) paired and permute across epochs
        RNG::manual_seed(2026);
        DataLoader cuda_loader(ds, batch_size, true, false, Device::CUDA);

        auto collect_epoch = [&](std::vector<size_t>& order_out) {
            order_out.clear();
            std::vector<size_t> seen(N, 0);
            for (auto& [bx, by] : cuda_loader) {
                CHECK_TENSOR(bx->device == Device::CUDA);
                CHECK_TENSOR(by->device == Device::CUDA);
                bx->to(Device::CPU);
                by->to(Device::CPU);

                for (size_t b = 0; b < bx->shape[0]; ++b) {
                    size_t label = static_cast<size_t>(std::llround((*by->data)[b]));
                    CHECK_TENSOR(label < N);
                    seen[label]++;
                    order_out.push_back(label);
                    for (size_t k = 0; k < sample_elems; ++k) {
                        double expected = static_cast<double>(label) * sample_tag_scale + static_cast<double>(k);
                        CHECK_CLOSE((*bx->data)[b * sample_elems + k], expected, strict_tol);
                    }
                }
            }
            for (size_t i = 0; i < N; ++i) {
                CHECK_EQUAL(seen[i], size_t{1});
            }
        };

        std::vector<size_t> epoch1_order, epoch2_order;
        collect_epoch(epoch1_order);
        collect_epoch(epoch2_order);
        CHECK_TENSOR(epoch1_order != epoch2_order);

        // multi-epoch vram leak check after cuda allocator warmup
        constexpr size_t leak_N = 256;
        constexpr size_t leak_elems = 4 * 16 * 16;
        constexpr size_t leak_bs = 64;
        constexpr int leak_epochs = 20;

        auto leak_x = std::make_shared<Tensor>(
            generate_4d_data(leak_N * leak_elems), std::vector<size_t>{leak_N, 4, 16, 16}, false, Device::CPU
        );
        auto leak_y = std::make_shared<Tensor>(
            std::vector<double>(leak_N, 1.0), std::vector<size_t>{leak_N}, false, Device::CPU
        );
        auto leak_ds = std::make_shared<TensorDataset>(leak_x, leak_y);
        DataLoader leak_loader(leak_ds, leak_bs, true, false, Device::CUDA);

        {
            for (auto& [bx, by] : leak_loader) {
                CHECK_TENSOR(bx->device == Device::CUDA);
                CHECK_TENSOR(by->device == Device::CUDA);
            }
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        size_t free_before = 0, total_mem = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_before, &total_mem));

        {
            for (int ep = 0; ep < leak_epochs; ++ep) {
                for (auto& [bx, by] : leak_loader) {
                    CHECK_EQUAL(bx->shape[0], leak_bs);
                    CHECK_EQUAL(by->shape[0], leak_bs);
                }
            }
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        size_t free_after = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_after, &total_mem));
        CHECK_TENSOR(free_after >= free_before);

        std::cout << "[PASS] Dataset & DataLoader shuffling, CUDA collation & multi-epoch zero-leak verified\n";
    }

    std::cout << "==========================================\n";
    std::cout << "[PASS] all GPU tests and latency benchmarks verified cleanly!\n";
    std::cout << "==========================================\n";

    return 0;
}