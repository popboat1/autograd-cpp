#ifndef DROPOUT_H
#define DROPOUT_H

#include "nn/Module.h"
#include "autograd/Tensor.h"

class Dropout : public Module
{
public:
    double p;
    explicit Dropout(double p=0.5);
    TensorPtr forward(const TensorPtr& input) override;
};

// dropout for 2D modules
class Dropout2D : public Module
{
public:
    double p;
    explicit Dropout2D(double p=0.5);
    TensorPtr forward(const TensorPtr& input) override;
};

# endif