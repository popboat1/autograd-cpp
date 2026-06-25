#ifndef VALUE_H
#define VALUE_H

#include <vector>
#include <memory>
#include <functional>
#include <set>
#include <string>
#include <iostream>

class Value;
using ValuePtr = std::shared_ptr<Value>;

class Value: public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad;
    std::string op;

    bool requires_grad = true;

    std::set<ValuePtr> prev; // set of pointers to parent nodes that created this Value

    std::function<void()> backward_func; // lambda function to pass grads back to parents

    //constructors
    Value(double val, bool requires_grad=true);
    Value(double val, std::set<ValuePtr> children, std::string operation="");

    void backward();
    
    // lists of operators
    friend ValuePtr operator+(const ValuePtr& lhs, const ValuePtr& rhs);
    friend ValuePtr operator*(const ValuePtr& lhs, const ValuePtr& rhs);
    friend ValuePtr operator/(const ValuePtr& lhs, const ValuePtr& rhs);
    friend ValuePtr operator-(const ValuePtr& lhs, const ValuePtr& rhs);
    ValuePtr pow(double exponent); // declared without friend because I am using this inside Value.cpp
    ValuePtr log();

    // activation functions
    ValuePtr tanh();
    ValuePtr exp();
    ValuePtr relu();

    void print() const;
};

inline ValuePtr make_val(double val, bool requires_grad=true){
    return std::make_shared<Value>(val, requires_grad);
}

#endif