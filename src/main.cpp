#include <iostream>
#include "autograd/Value.h"

int main() {
    auto a = make_val(-2.0);
    auto b = make_val(3.0);
    auto c = make_val(10.0);

    // simple forward pass of L = (a * b) + c
    auto x = a * b;
    auto L = x + c;

    // backward pass
    L->backward();

    std::cout << "results\n";
    std::cout << "forward pass (L = (a * b) + c): " << L->data << "\n";
    std::cout << "gradient of a: " << a->grad << "\n";
    std::cout << "gradient of b: " << b->grad << "\n";
    std::cout << "gradient of c: " << c->grad << "\n";
    
    return 0;
}
