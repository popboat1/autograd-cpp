#include <iostream>
#include <cassert>
#include <cmath>
#include "autograd/Value.h"

// helper function to check if floating point numbers are close enough...
bool is_close(double a, double b, double tolerance = 1e-5) {
    return std::abs(a - b) < tolerance;
}

void test_basic_math(){
    auto a = make_val(2.0);
    auto b = make_val(3.0);

    // L = a * b + b^2 -> 2 * 3 + 3^2 = 15
    auto L = (a * b) + b->pow(2);
    L->backward();

    // check forward pass
    assert(is_close(L->data, 15.0));

    // check backward pass gradients
    // dL/da = b = 3
    assert(is_close(a->grad, 3.0));
    // dL/db = a + 2*b = 2 + 2(3) = 8
    assert(is_close(b->grad, 8.0));

    std::cout << "[PASS] Basic Scalar Math & Gradient Test\n";
}

void test_subtraction_and_division() {
    auto a = make_val(10.0);
    auto b = make_val(2.0);
    
    auto L = (a - b) / b; // (10 - 2) / 2 = 4.0
    L->backward();

    assert(is_close(L->data, 4.0));
    std::cout << "[PASS] Subtraction & Division Test\n";
}

int main() {
    std::cout << "running test...\n";
    test_basic_math();
    test_subtraction_and_division();
    std::cout << "[DONE] test is done!!1!\n";
    return 0;
}
