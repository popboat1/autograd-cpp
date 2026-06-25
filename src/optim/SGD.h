#ifndef SGD_H
#define SGD_H

#include "autograd/Tensor.h"
#include <vector>

class SGD {
public:
    // pass model's params and lr
    SGD(std::vector<TensorPtr> params, double lr, double momentum = 0.0, double weight_decay = 0.0);

    void step();
    void zero_grad();

private:
    std::vector<TensorPtr> params;
    std::vector<std::vector<double>> velocities;
    double lr;
    double momentum_factor;
    double wd;
};

#endif