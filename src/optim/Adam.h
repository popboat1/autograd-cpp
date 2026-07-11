#ifndef ADAM_H
#define ADAM_H

#include "autograd/Tensor.h"
#include <vector>
#include <utility>

class Adam {
public:
    std::vector<TensorPtr> params;
    double lr;
    double beta1;
    double beta2;
    double eps;
    double weight_decay;
    bool amsgrad;
    bool maximize;

    size_t t; // step counter for bias correction exponents

    // state tracking storage vectors
    std::vector<TensorPtr> exp_avgs;        // first moment vectors (m_t)
    std::vector<TensorPtr> exp_avg_sqs;     // second moment vectors (v_t)
    std::vector<TensorPtr> max_exp_avg_sqs;  // maximum second moment vectors (v_max_t)

    Adam(std::vector<TensorPtr> params, double lr = 0.001, 
         std::pair<double, double> betas = {0.9, 0.999}, 
         double eps = 1e-8, double weight_decay = 0.0, 
         bool amsgrad = false, bool maximize = false);

    void step();
    void zero_grad();
};

#endif