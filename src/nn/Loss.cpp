#include "Loss.h"
#include <stdexcept>
#include <cmath>

// mean squared loss
ValuePtr MSELoss::operator()(const std::vector<ValuePtr>& preds, const std::vector<ValuePtr>& targets){
    if(preds.size() != targets.size()){
        throw std::invalid_argument("MSELoss: predictions and targets must have same dimensions!");
    }

    auto total_loss = make_val(0.0);
    for(size_t i {0}; i < preds.size(); ++i){
        auto diff = preds[i] - targets[i];
        total_loss = total_loss + diff->pow(2);
    }

    auto N = make_val(static_cast<double>(preds.size()));
    return total_loss/N;
}

// CrossEntropyLoss: softmax + neg-log-likelihood over raw logits
ValuePtr CrossEntropyLoss::operator()(const std::vector<ValuePtr>& logits, int target_idx){
    if (target_idx < 0 || target_idx >= static_cast<int>(logits.size())) {
        throw std::out_of_range("CrossEntropyLoss: target index out of bounds!");
    }

    auto sum_exp = make_val(0.0);
    for (const auto& logit : logits) {
        sum_exp = sum_exp + logit->exp();
    }

    // LogSumExp(logits) - logit[target]
    // mathematically identical to -log(exp(logit_target) / sum(exp(logits)))
    ValuePtr loss = sum_exp->log() - logits[target_idx];
    
    return loss;
}