#include "Loss.h"
#include <stdexcept>
#include <cmath>

// mean squared loss
TensorPtr MSELoss::operator()(const TensorPtr& preds, const TensorPtr& targets){
    if (preds->shape != targets->shape) {
        throw std::invalid_argument("MSELoss: predictions and targets must have same dimensions!");
    }

    auto diff = preds - targets;
    auto squared = diff->pow(2.0);

    // sum across the batch
    auto sum_tensor = squared->sum();

    // multiply by 1/N to get the mean
    double N = static_cast<double>(preds->data->size());
    auto N_tensor = std::make_shared<Tensor>(std::vector<double>{1.0 / N}, std::vector<size_t>{1}, false);
    
    return sum_tensor * N_tensor;
}

// will be updated later once softmax has been introduced into tensor class
// CrossEntropyLoss: softmax + neg-log-likelihood over raw logits
// TensorPtr CrossEntropyLoss::operator()(const TensorPtr& logits, int target_idx){
//     if (target_idx < 0 || target_idx >= static_cast<int>(logits->size())) {
//         throw std::out_of_range("CrossEntropyLoss: target index out of bounds!");
//     }

//     auto sum_exp = make_val(0.0);
//     for (const auto& logit : logits) {
//         sum_exp = sum_exp + logit->exp();
//     }

//     // LogSumExp(logits) - logit[target]
//     // mathematically identical to -log(exp(logit_target) / sum(exp(logits)))
//     ValuePtr loss = sum_exp->log() - logits[target_idx];
    
//     return loss;
// }