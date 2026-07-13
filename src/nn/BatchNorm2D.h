#ifndef BATCHNORM2D_H
#define BATCHNORM2D_H

#include "nn/Module.h"
#include "autograd/Tensor.h"
#include <vector>
#include <string>

class BatchNorm2D : public Module {
public:
    size_t num_features;
    double eps;
    double momentum;
    bool affine;
    bool track_running_stats;
    bool training; // track module runtime mode state (training vs evaluation)

    // differentiable tracking parameters
    TensorPtr weight; // gamma scaling factor
    TensorPtr bias;   // beta shifting factor

    // non-differentiable historical tracking buffers
    TensorPtr running_mean;
    TensorPtr running_var;

    BatchNorm2D(size_t num_features, double eps = 1e-5, double momentum = 0.1, 
                bool affine = true, bool track_running_stats = true);

    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() const override;
};

#endif