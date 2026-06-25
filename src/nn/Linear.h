#ifndef LINEAR_H
#define LINEAR_H

#include "Module.h"

class Linear : public Module {
public:
    Linear(int fan_in, int fan_out, int seed=42); // init weights and biases randomly based on in/out size

    // forward pass
    TensorPtr forward(const TensorPtr& xin);
    
    // collects all weight and bias scalar pointers into a flat list
    std::vector<TensorPtr> parameters() const override;

private:
    int fan_in;
    int fan_out;
    TensorPtr weights; // [fan_in][fan_out]
    TensorPtr biases; // [fanout]
};

#endif