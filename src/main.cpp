#include <iostream>
#include "autograd/Value.h"

int main() {
    auto a = make_val(-2.0);
    auto b = make_val(3.0);
    auto c = make_val(10.0);

    // forward pass
    auto x1 = a * b;
    auto x2 = x1 + c;
    auto x3 = x2 - a;
    auto x4 = x3 / b;

    // activations
    auto x5 = x4->tanh();
    auto x6 = x5->exp();
    auto x7 = x6->relu();

    auto L  = x7->pow(2);

    // backward pass
    L->backward();

    std::cout << "results\n";
    std::cout << "L: " << L->data << "\n";
    std::cout << "da: " << a->grad << "\n";
    std::cout << "db: " << b->grad << "\n";
    std::cout << "dc: " << c->grad << "\n";
    
    return 0;
}
