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

// crossentropyloss
TensorPtr CrossEntropyLoss::operator()(const TensorPtr& logits, const TensorPtr& targets){
    if (logits->shape != targets->shape) {
        throw std::invalid_argument("CrossEntropyLoss: logits and one-hot targets must have same dimensions!");
    }
    
    auto max_logits = logits->max(1, true);                // extract maximum logits along class dim (axis 1)
    auto shifted_logits = logits - max_logits;             // compute shifted logits to prevent exponent overflow
    auto exp_logits = shifted_logits->exp();               // exponentiate shifted logits
    auto sum_exp = exp_logits->sum(1, true);               // sum exponents along the class dimension
    auto log_sum_exp = sum_exp->log();                     // compute the logarithm of the sum of exponents
    auto lse = log_sum_exp + max_logits;                   // complete LogSumExp restoration
    auto target_logits = (logits * targets)->sum(1, true); // extract target logits via element-wise multiplication with one-hot targets
    auto loss_per_sample = lse - target_logits;            // per-sample loss calculation

    // flatten and average the loss across the batch elements
    auto total_loss = loss_per_sample->sum();
    double N = static_cast<double>(logits->shape[0]); // normalize by batch size
    auto N_tensor = std::make_shared<Tensor>(std::vector<double>{1.0 / N}, std::vector<size_t>{1}, false);

    return total_loss * N_tensor;
}

// sparse categorical cross entropy loss
TensorPtr SparseCategoricalCrossEntropyLoss::operator()(const TensorPtr& logits, const TensorPtr& targets) {
    if (logits->shape.size() < 2) {
        throw std::invalid_argument("sparse categorical cross entropy loss: logits must be at least 2d!");
    }
    
    size_t batch_size = logits->shape[0];
    size_t num_classes = logits->shape[1];
    
    if (targets->data->size() != batch_size) {
        throw std::invalid_argument("sparse categorical cross entropy loss: targets size must match batch size!");
    }
    
    // compute stable log softmax probabilities using the compositional op
    auto log_probs = logits->log_softmax(1);
    
    double loss_sum = 0.0;
    std::vector<size_t> target_indices(batch_size);
    
    // extract log probabilities corresponding to the integer target indices
    for (size_t b = 0; b < batch_size; ++b) {
        size_t target_idx = static_cast<size_t>((*targets->data)[b]);
        if (target_idx >= num_classes) {
            throw std::out_of_range("sparse categorical cross entropy loss: target index out of bounds!");
        }
        target_indices[b] = target_idx;
        loss_sum += (*log_probs->data)[b * num_classes + target_idx];
    }
    
    // compute negative mean loss scalar
    double final_loss = -loss_sum / static_cast<double>(batch_size);
    
    // pack into a new tracking graph node depending on the log_probs tensor
    auto out = std::make_shared<Tensor>(std::vector<double>{final_loss}, std::vector<size_t>{1}, std::vector<TensorPtr>{log_probs}, "sparse_categorical_cross_entropy");
    
    // backward pass routes upstream loss scale back to selected target class nodes
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [log_probs, target_indices, batch_size, num_classes, weak_out]() {
        if (auto out_ptr = weak_out.lock()) {
            if (log_probs->requires_grad) {
                double upstream_grad = (*out_ptr->grad)[0];
                double grad_scale = -1.0 / static_cast<double>(batch_size);
                for (size_t b = 0; b < batch_size; ++b) {
                    size_t flat_idx = b * num_classes + target_indices[b];
                    (*log_probs->grad)[flat_idx] += upstream_grad * grad_scale;
                }
            }
        }
    };
    
    return out;
}