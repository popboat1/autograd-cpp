#include "Value.h"
#include <cmath>

Value::Value(double val) : data(val), grad(0.0), op(""), backward_func([](){}) {}

Value::Value(double val, std::set<ValuePtr> children, std::string operation)
    : data(val), grad(0.0), prev(children), op(operation), backward_func([](){}){}

// addition op
ValuePtr operator+(const ValuePtr& lhs, const ValuePtr& rhs){
    auto out = std::make_shared<Value>(
        lhs->data + rhs->data,
        std::set<ValuePtr>{lhs, rhs}, "+"
    );

    //backward pass
    out->backward_func = [lhs, rhs, out](){
        lhs->grad += out->grad;
        rhs->grad += out->grad;
    };
    return out;
}

// multiplication op
ValuePtr operator*(const ValuePtr& lhs, const ValuePtr& rhs){
    auto out = std::make_shared<Value>(
        lhs->data * rhs->data,
        std::set<ValuePtr>{lhs, rhs}, "*"
    );

    // dL/dx = y * dL/dout
    out->backward_func = [lhs, rhs, out]() {
        lhs->grad += rhs->data * out->grad;
        rhs->grad += lhs->data * out->grad;
    };
    return out;
}

// substraction op
ValuePtr operator-(const ValuePtr& lhs, const ValuePtr& rhs){
    return lhs + (rhs * make_val(-1.0));
}

// exponents op
ValuePtr Value::pow(double exponent){
    double out_data = std::pow(this->data, exponent);

    auto out = std::make_shared<Value>(
        out_data,
        std::set<ValuePtr>{shared_from_this()},
        "pow(" + std::to_string(exponent) + ")"
    );

    // backward pass
    // dL/dx = exponent * base^(exponent - 1)
    auto self = shared_from_this();
    out->backward_func = [self, exponent, out](){
        double local_derivative = exponent * std::pow(self->data, exponent - 1);

        self->grad += local_derivative * out->grad;
    };

    return out;
}

// division op
ValuePtr operator/(const ValuePtr& lhs, const ValuePtr& rhs){
    return lhs * rhs->pow(-1.0);
}

// topological sort to execute backward pass sequentially
void Value::backward(){
    std::vector<ValuePtr> topo;
    std::set<ValuePtr> visited;

    std::function<void(ValuePtr)> build_topo = [&](ValuePtr v) {
        if(visited.find(v) == visited.end()){
            visited.insert(v);
            for(const auto& child : v->prev){
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(shared_from_this());

    this->grad = 1.0; // out node start with grad 1.0

    // process nodes in reverse topo order
    for(auto it = topo.rbegin(); it != topo.rend(); ++it){
        (*it)->backward_func();
    }
}

void Value::print() const {
    std::cout << "Value(data=" << data << ", grad=" << grad << ", op='" << op << "')\n";
}
