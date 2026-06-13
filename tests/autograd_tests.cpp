#include <iostream>
#include <cassert>
#include <cmath>
#include "autograd/Value.h"
#include "nn/Linear.h"

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

void test_requires_grad(){
    auto a = make_val(2.0, true);
    auto b = make_val(2.0, false);

    // propagation on a completely frozen branch
    auto c = b * b;
    assert(c->requires_grad == false); // both children false -> false

    // propagation on a mixed branch
    auto loss = a + c;
    assert(loss->requires_grad == true); // 1 child true -> true

    loss->backward();

    assert(is_close(a->grad, 1.0));
    assert(is_close(b->grad, 0.0));

    std::cout << "[PASS] Requires Grad Tests\n";
}

void test_subtraction_and_division() {
    auto a = make_val(10.0);
    auto b = make_val(2.0);
    
    auto L = (a - b) / b; // (10 - 2) / 2 = 4.0
    L->backward();

    assert(is_close(L->grad, 1.0));
    assert(is_close(L->data, 4.0));

    // dL/da = 1/b = 1/2 = 0.5
    assert(is_close(a->grad, 0.5));
    // dL/db = -a/(b^2) = -10/4 = -2.5
    assert(is_close(b->grad, -2.5));
    std::cout << "[PASS] Subtraction & Division Test\n";
}

void test_activation_func() {
    // tanh()
    auto a = make_val(2.0);
    auto L_tanh = a->tanh();
    L_tanh->backward();

    // tanh(2.0) ≈ 0.96403
    assert(is_close(L_tanh->data, 0.96402758));
    // d/dx tanh(2.0) = 1 - tanh^2(2.0) ≈ 0.07065
    assert(is_close(a->grad, 0.07065082));

    // exp()
    auto b = make_val(1.0);
    auto L_exp = b->exp();
    L_exp->backward();
    
    // exp(1.0) ≈ 2.71828
    assert(is_close(L_exp->data, 2.71828182));
    // d/dx exp(1.0) = exp(1.0) ≈ 2.71828
    assert(is_close(b->grad, 2.71828182));

    // relu() positive boundary
    auto c = make_val(1.5);
    auto L_relu1 = c->relu();
    L_relu1->backward();
    
    assert(is_close(L_relu1->data, 1.5));
    assert(is_close(c->grad, 1.0)); // slope is 1.0 for x > 0

    // relu() negative boundary
    auto d = make_val(-3.0);
    auto L_relu2 = d->relu();
    L_relu2->backward();
    
    assert(is_close(L_relu2->data, 0.0)); // clamped to 0
    assert(is_close(d->grad, 0.0)); // slope flatlines to 0.0 for x <= 0

    std::cout << "[PASS] Activation Functions (tanh, exp, relu) Test\n";
}

void test_linear_layer(){
    int fan_in = 3;
    int fan_out = 2;
    Linear layer(fan_in, fan_out, 42); // seed 42 for reproducibility

    // test parameter management
    auto params = layer.parameters();
    assert(params.size() == 8);
    
    // test forward pass
    std::vector<ValuePtr> xin = { make_val(1.0), make_val(-2.0), make_val(0.5) };
    auto xout = layer.forward(xin);
    assert(xout.size() == 2);

    // test backward pass
    auto loss = xout[0] + xout[1];
    loss->backward();

    // check computed gradients
    assert(xin[0]->grad != 0.0);
    assert(xin[1]->grad != 0.0);
    assert(xin[2]->grad != 0.0);

    // test zero_grad function
    layer.zero_grad();
    for (const auto& p : layer.parameters()) {
        assert(p->grad == 0.0);
    }

    std::cout << "[PASS] NN Linear Layer Structural Test\n";
}

int main() {
    std::cout << "running test...\n";
    test_basic_math();
    test_subtraction_and_division();
    test_activation_func();
    test_requires_grad();
    test_linear_layer();
    std::cout << "[DONE] test is done!!1!\n";
    return 0;
}
