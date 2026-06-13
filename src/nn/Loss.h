#ifndef LOSS_H
#define LOSS_H

#include "autograd/Value.h"
#include <vector>

class MSELoss {
public:
    MSELoss() = default;
    ValuePtr operator()(const std::vector<ValuePtr>& preds, const std::vector<ValuePtr>& targets);
};

class CrossEntropyLoss {
public:
    CrossEntropyLoss() = default;
    ValuePtr operator()(const std::vector<ValuePtr>& logits, int target_idx);
};

#endif