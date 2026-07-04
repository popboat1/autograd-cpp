#ifndef LOSS_H
#define LOSS_H

#include "autograd/Tensor.h"
#include <vector>

class MSELoss {
public:
    MSELoss() = default;
    TensorPtr operator()(const TensorPtr& preds, const TensorPtr& targets);
};

class CrossEntropyLoss {
public:
    CrossEntropyLoss() = default;
    TensorPtr operator()(const TensorPtr& logits, const TensorPtr& targets);
};

#endif