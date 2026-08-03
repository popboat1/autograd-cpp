#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "autograd/Tensor.h"

// helper to assert floating point parity smoothly
bool close_enough(double a, double b, double tol = 1e-4) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << "starting tensorgrad gpu unary tests\n";

    // test 1: relu forward and backward on gpu
    {
        auto x = std::make_shared<Tensor>(std::vector<double>{-2.0, 0.0, 3.0}, std::vector<size_t>{3}, true);
        x->to(Device::CUDA);
        auto out = x->relu();
        
        // copy to cpu to verify forward calculations
        out->to(Device::CPU);
        assert(close_enough((*out->data)[0], 0.0));
        assert(close_enough((*out->data)[1], 0.0));
        assert(close_enough((*out->data)[2], 3.0));

        // push back to gpu and trigger backward pass
        out->to(Device::CUDA);
        out->backward();
        x->to(Device::CPU);

        assert(close_enough((*x->grad)[0], 0.0));
        assert(close_enough((*x->grad)[1], 0.0));
        assert(close_enough((*x->grad)[2], 1.0));
        std::cout << "[PASS] relu cuda pass verified\n";
    }

    // test 2: exp forward and backward on gpu
    {
        auto x = std::make_shared<Tensor>(std::vector<double>{0.0, 1.0, 2.0}, std::vector<size_t>{3}, true);
        x->to(Device::CUDA);
        auto out = x->exp();

        out->to(Device::CPU);
        assert(close_enough((*out->data)[0], 1.0));
        assert(close_enough((*out->data)[1], std::exp(1.0)));
        assert(close_enough((*out->data)[2], std::exp(2.0)));

        out->to(Device::CUDA);
        out->backward();
        x->to(Device::CPU);

        assert(close_enough((*x->grad)[0], 1.0));
        assert(close_enough((*x->grad)[1], std::exp(1.0)));
        assert(close_enough((*x->grad)[2], std::exp(2.0)));
        std::cout << "[PASS] exp cuda pass verified\n";
    }

    // test 3: tanh forward and backward on gpu
    {
        auto x = std::make_shared<Tensor>(std::vector<double>{0.0, 0.5}, std::vector<size_t>{2}, true);
        x->to(Device::CUDA);
        auto out = x->tanh();

        out->to(Device::CPU);
        double t0 = std::tanh(0.0);
        double t1 = std::tanh(0.5);
        assert(close_enough((*out->data)[0], t0));
        assert(close_enough((*out->data)[1], t1));

        out->to(Device::CUDA);
        out->backward();
        x->to(Device::CPU);

        assert(close_enough((*x->grad)[0], 1.0 - t0 * t0));
        assert(close_enough((*x->grad)[1], 1.0 - t1 * t1));
        std::cout << "[PASS] tanh cuda pass verified\n";
    }

    // test 4: sigmoid forward and backward on gpu
    {
        auto x = std::make_shared<Tensor>(std::vector<double>{0.0, 2.0}, std::vector<size_t>{2}, true);
        x->to(Device::CUDA);
        auto out = x->sigmoid();

        out->to(Device::CPU);
        double s0 = 1.0 / (1.0 + std::exp(0.0));
        double s1 = 1.0 / (1.0 + std::exp(-2.0));
        assert(close_enough((*out->data)[0], s0));
        assert(close_enough((*out->data)[1], s1));

        out->to(Device::CUDA);
        out->backward();
        x->to(Device::CPU);

        assert(close_enough((*x->grad)[0], s0 * (1.0 - s0)));
        assert(close_enough((*x->grad)[1], s1 * (1.0 - s1)));
        std::cout << "[PASS] sigmoid cuda pass verified\n";
    }

    // test 5: log forward and backward on gpu
    {
        auto x = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 10.0}, std::vector<size_t>{3}, true);
        x->to(Device::CUDA);
        auto out = x->log();

        out->to(Device::CPU);
        assert(close_enough((*out->data)[0], 0.0));
        assert(close_enough((*out->data)[1], std::log(2.0)));
        assert(close_enough((*out->data)[2], std::log(10.0)));

        out->to(Device::CUDA);
        out->backward();
        x->to(Device::CPU);

        assert(close_enough((*x->grad)[0], 1.0));
        assert(close_enough((*x->grad)[1], 0.5));
        assert(close_enough((*x->grad)[2], 0.1));
        std::cout << "[PASS] log cuda pass verified\n";
    }

    // test 6: pow forward and backward on gpu
    {
        auto x = std::make_shared<Tensor>(std::vector<double>{2.0, 3.0}, std::vector<size_t>{2}, true);
        x->to(Device::CUDA);
        auto out = x->pow(3.0);

        out->to(Device::CPU);
        assert(close_enough((*out->data)[0], 8.0));
        assert(close_enough((*out->data)[1], 27.0));

        out->to(Device::CUDA);
        out->backward();
        x->to(Device::CPU);

        // d(x^3)/dx = 3 * x^2
        assert(close_enough((*x->grad)[0], 12.0));
        assert(close_enough((*x->grad)[1], 27.0));
        std::cout << "[PASS] pow cuda pass verified\n";
    }

    // test 7: sqrt forward and backward on gpu
    {
        auto x = std::make_shared<Tensor>(std::vector<double>{4.0, 16.0}, std::vector<size_t>{2}, true);
        x->to(Device::CUDA);
        auto out = x->sqrt();

        out->to(Device::CPU);
        assert(close_enough((*out->data)[0], 2.0));
        assert(close_enough((*out->data)[1], 4.0));

        out->to(Device::CUDA);
        out->backward();
        x->to(Device::CPU);

        // d(sqrt(x))/dx = 0.5 / sqrt(x)
        assert(close_enough((*x->grad)[0], 0.25));
        assert(close_enough((*x->grad)[1], 0.125));
        std::cout << "[PASS] sqrt cuda pass verified\n";
    }

    // test 8: neg / prefix operator- forward and backward on gpu
    {
        auto x = std::make_shared<Tensor>(std::vector<double>{1.5, -4.0}, std::vector<size_t>{2}, true);
        x->to(Device::CUDA);
        auto out = -x;

        out->to(Device::CPU);
        assert(close_enough((*out->data)[0], -1.5));
        assert(close_enough((*out->data)[1], 4.0));

        out->to(Device::CUDA);
        out->backward();
        x->to(Device::CPU);

        assert(close_enough((*x->grad)[0], -1.0));
        assert(close_enough((*x->grad)[1], -1.0));
        std::cout << "[PASS] neg cuda pass verified\n";
    }
    std::cout << "[PASS] all unary kernels verified\n";

    // test 9: addition matching shape fast path on gpu
    {
        auto a = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0}, std::vector<size_t>{3}, true);
        auto b = std::make_shared<Tensor>(std::vector<double>{4.0, 5.0, 6.0}, std::vector<size_t>{3}, true);
        a->to(Device::CUDA);
        b->to(Device::CUDA);

        auto out = a + b;

        out->to(Device::CPU);
        assert(close_enough((*out->data)[0], 5.0));
        assert(close_enough((*out->data)[1], 7.0));
        assert(close_enough((*out->data)[2], 9.0));

        out->to(Device::CUDA);
        out->backward();
        a->to(Device::CPU);
        b->to(Device::CPU);

        assert(close_enough((*a->grad)[0], 1.0));
        assert(close_enough((*a->grad)[2], 1.0));
        assert(close_enough((*b->grad)[0], 1.0));
        assert(close_enough((*b->grad)[2], 1.0));
        std::cout << "[PASS] operator+ cuda fast path verified\n";
    }

    // test 10: addition broadcasting path on gpu
    {
        auto a = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, std::vector<size_t>{2, 3}, true);
        auto b = std::make_shared<Tensor>(std::vector<double>{10.0, 20.0, 30.0}, std::vector<size_t>{3}, true);
        a->to(Device::CUDA);
        b->to(Device::CUDA);

        auto out = a + b;

        out->to(Device::CPU);
        assert(close_enough((*out->data)[0], 11.0));
        assert(close_enough((*out->data)[2], 33.0));
        assert(close_enough((*out->data)[3], 14.0));
        assert(close_enough((*out->data)[5], 36.0));

        out->to(Device::CUDA);
        out->backward();
        a->to(Device::CPU);
        b->to(Device::CPU);

        // a receives 1.0 for each element
        assert(close_enough((*a->grad)[0], 1.0));
        assert(close_enough((*a->grad)[5], 1.0));

        // b is broadcasted across 2 rows, so gradients accumulate to 2.0
        assert(close_enough((*b->grad)[0], 2.0));
        assert(close_enough((*b->grad)[1], 2.0));
        assert(close_enough((*b->grad)[2], 2.0));
        std::cout << "[PASS] operator+ cuda broadcast path verified\n";
    }

    // test 11: subtraction matching shape fast path on gpu
    {
        auto a = std::make_shared<Tensor>(std::vector<double>{5.0, 8.0, 12.0}, std::vector<size_t>{3}, true);
        auto b = std::make_shared<Tensor>(std::vector<double>{2.0, 3.0, 4.0}, std::vector<size_t>{3}, true);
        a->to(Device::CUDA);
        b->to(Device::CUDA);

        auto out = a - b;

        out->to(Device::CPU);
        assert(close_enough((*out->data)[0], 3.0));
        assert(close_enough((*out->data)[1], 5.0));
        assert(close_enough((*out->data)[2], 8.0));

        out->to(Device::CUDA);
        out->backward();
        a->to(Device::CPU);
        b->to(Device::CPU);

        assert(close_enough((*a->grad)[0], 1.0));
        assert(close_enough((*a->grad)[2], 1.0));
        assert(close_enough((*b->grad)[0], -1.0));
        assert(close_enough((*b->grad)[2], -1.0));
        std::cout << "[PASS] operator- cuda fast path verified\n";
    }

    // test 12: subtraction broadcasting path on gpu
    {
        auto a = std::make_shared<Tensor>(std::vector<double>{10.0, 20.0, 30.0, 40.0, 50.0, 60.0}, std::vector<size_t>{2, 3}, true);
        auto b = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0}, std::vector<size_t>{3}, true);
        a->to(Device::CUDA);
        b->to(Device::CUDA);

        auto out = a - b;

        out->to(Device::CPU);
        assert(close_enough((*out->data)[0], 9.0));
        assert(close_enough((*out->data)[2], 27.0));
        assert(close_enough((*out->data)[3], 39.0));
        assert(close_enough((*out->data)[5], 57.0));

        out->to(Device::CUDA);
        out->backward();
        a->to(Device::CPU);
        b->to(Device::CPU);

        // a receives +1.0 for each element
        assert(close_enough((*a->grad)[0], 1.0));
        assert(close_enough((*a->grad)[5], 1.0));

        // b is broadcasted across 2 rows, accumulating -1.0 twice = -2.0
        assert(close_enough((*b->grad)[0], -2.0));
        assert(close_enough((*b->grad)[1], -2.0));
        assert(close_enough((*b->grad)[2], -2.0));
        std::cout << "[PASS] operator- cuda broadcast path verified\n";
    }

    std::cout << "[PASS] all gpu tests passed cleanly\n";
    return 0;
}