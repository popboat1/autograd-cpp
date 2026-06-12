#include "Value.h"
#include <cmath>
#include <algorithm>

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
    std::weak_ptr<Value> weak_out = out; // capture to break the circular reference cycle
                                         // (out -> backward_func -> out) and prevent memory leaks.
    out->backward_func = [lhs, rhs, weak_out](){
        if(auto out_ptr = weak_out.lock()){
            lhs->grad += out_ptr->grad;
            rhs->grad += out_ptr->grad;
        }
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
    std::weak_ptr<Value> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out]() {
        if(auto out_ptr = weak_out.lock()){
            lhs->grad += rhs->data * out_ptr->grad;
            rhs->grad += lhs->data * out_ptr->grad;
        }
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
    std::weak_ptr<Value> weak_out = out;
    out->backward_func = [self, exponent, weak_out](){
        if(auto out_ptr = weak_out.lock()){
            double local_derivative = exponent * std::pow(self->data, exponent - 1);

            self->grad += local_derivative * out_ptr->grad;
        }
    };

    return out;
}

// division op
ValuePtr operator/(const ValuePtr& lhs, const ValuePtr& rhs){
    return lhs * rhs->pow(-1.0);
}

// tanh function
ValuePtr Value::tanh(){
    auto out_data = std::tanh(this->data);

    auto out = std::make_shared<Value>(
        out_data,
        std::set<ValuePtr>{shared_from_this()},
        "tanh(" + std::to_string(this->data) + ")"
    );

    // backward pass
    // d/dx tanh(x) = 1 - tanh^2(x)
    auto self = shared_from_this();
    std::weak_ptr<Value> weak_out = out;
    out->backward_func = [self, out_data, weak_out](){
        if(auto out_ptr = weak_out.lock()){
            double local_derivative = 1.0 - std::pow(out_data, 2);

            self->grad += local_derivative * out_ptr->grad;
        }

    };

    return out;
}

// exp function
ValuePtr Value::exp(){
    auto out_data = std::exp(this->data);

    auto out = std::make_shared<Value>(
        out_data,
        std::set<ValuePtr>{shared_from_this()},
        "exp(" + std::to_string(this->data) + ")"
    );

    // backward pass
    // d/dx exp(x) = exp(x)
    auto self = shared_from_this();
    std::weak_ptr<Value> weak_out = out;
    out->backward_func = [self, out_data, weak_out](){
        if(auto out_ptr = weak_out.lock()){
            self->grad += out_ptr->data * out_ptr->grad;
        }
    };

    return out;
}

// relu function
ValuePtr Value::relu(){
    auto out_data = std::max(0.0, this->data);

    auto out = std::make_shared<Value>(
        out_data,
        std::set<ValuePtr>{shared_from_this()},
        "relu(" + std::to_string(this->data) + ")"
    );

    // backward pass
    // d/dx relu() -> 1.0 if positive otherwise 0.0
    auto self = shared_from_this();
    std::weak_ptr<Value> weak_out = out;
    out->backward_func = [self, out_data, weak_out](){
        if(auto out_ptr = weak_out.lock()){
            double local_derivative = self->data > 0 ? 1.0 : 0.0;

            self->grad += local_derivative * out_ptr->grad;
        }
    };

    return out;
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
