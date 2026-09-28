#include "Adam.h"
#include <cmath>
#include <algorithm>
#include "utils/cuda_utils.h"

Adam::Adam(std::vector<TensorPtr> params, double lr, std::pair<double, double> betas, 
           double eps, double weight_decay, bool amsgrad, bool maximize)
    : params(params), lr(lr), beta1(betas.first), beta2(betas.second), 
      eps(eps), weight_decay(weight_decay), amsgrad(amsgrad), maximize(maximize), t(0) {
    
    // allocate zero-filled tracking moment buffers matching shapes of target parameters
    for (const auto& p : params) {
        size_t total_elements = p->data->size();
        exp_avgs.push_back(std::make_shared<Tensor>(std::vector<double>(total_elements, 0.0), p->shape, false, p->device));
        exp_avg_sqs.push_back(std::make_shared<Tensor>(std::vector<double>(total_elements, 0.0), p->shape, false, p->device));
        
        if (amsgrad) {
            max_exp_avg_sqs.push_back(std::make_shared<Tensor>(std::vector<double>(total_elements, 0.0), p->shape, false, p->device));
        } else {
            max_exp_avg_sqs.push_back(nullptr);
        }
    }
}

// cuda kernel
__global__ void d_adam_step(
    double* __restrict__ param,             // pointer to param weights
    const double* __restrict__ grad,        // pointer to grads
    double* __restrict__ exp_avg,           // first moment vector
    double* __restrict__ exp_avg_sq,        // second moment vector
    double* __restrict__ max_exp_avg_sq,    // max second moment
    size_t n,                               // total element
    double lr,                              // learning rate
    double beta1, double beta2,             // decay factors
    double eps,                             // epsilon
    double weight_decay,                    // weight decay penalty
    double bias_correction1,     
    double bias_correction2,
    bool maximize  // grad direction flag
){
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= n) return;

    // load grad
    double g = grad[idx];

    if(maximize) g = -g;

    // apply weight decay if rpesent
    if(weight_decay != 0.0){
        g += weight_decay * param[idx];
    }

    // update biased 1st moment
    double m = beta1 * exp_avg[idx] + (1.0 - beta1) * g;
    exp_avg[idx] = m;

    // update biased 2nd moment
    double v = beta2 * exp_avg_sq[idx] + (1 - beta2) * g * g;
    exp_avg_sq[idx] = v;

    // cmpute m_hat
    double m_hat = m / bias_correction1;
    double v_hat;

    // compute v_hat
    if(max_exp_avg_sq != nullptr){
        double v_max = fmax(max_exp_avg_sq[idx], v);
        max_exp_avg_sq[idx] = v_max;

        v_hat = v_max / bias_correction2;
    } else {
        v_hat = v / bias_correction2;
    }
    // update param
    param[idx] -= lr * m_hat / (sqrt(v_hat) + eps);
}

inline void launch_adam_step(
    double* __restrict__ param,             // pointer to param weights
    const double* __restrict__ grad,        // pointer to grads
    double* __restrict__ exp_avg,           // first moment vector
    double* __restrict__ exp_avg_sq,        // second moment vector
    double* __restrict__ max_exp_avg_sq,    // max second moment
    size_t n,                               // total element
    double lr,                              // learning rate
    double beta1, double beta2,             // decay factors
    double eps,                             // epsilon
    double weight_decay,                    // weight decay penalty
    double bias_correction1,     
    double bias_correction2,
    bool maximize  // grad direction flag
){
    constexpr size_t block_threads = 256;
    int blocks = cuda_utils::ceil_div(static_cast<int>(n), block_threads);
    d_adam_step<<<blocks, block_threads>>>(
        param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq, 
        n, lr, beta1, beta2, eps, weight_decay, 
        bias_correction1, bias_correction2, maximize
    );
    CUDA_CHECK(cudaGetLastError());
}

void Adam::step() {
    // increment historical step counter tracking for runtime bias corrections
    t++;
    
    double bias_correction1 = 1.0 - std::pow(beta1, t);
    double bias_correction2 = 1.0 - std::pow(beta2, t);

    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->device == Device::CUDA) {
            // skip if no gradients were accumulated on device
            if (params[i]->cuda_grad == nullptr) continue;
            // ensure moment tracking tensors reside on cuda
            if (exp_avgs[i]->device != Device::CUDA) exp_avgs[i]->to(Device::CUDA);
            if (exp_avg_sqs[i]->device != Device::CUDA) exp_avg_sqs[i]->to(Device::CUDA);
            if (amsgrad && max_exp_avg_sqs[i] && max_exp_avg_sqs[i]->device != Device::CUDA) {
                max_exp_avg_sqs[i]->to(Device::CUDA);
            }
            size_t total_elements = params[i]->data->size();
            double* d_param = params[i]->cuda_data.get();
            const double* d_grad = params[i]->cuda_grad.get();
            double* d_m = exp_avgs[i]->cuda_data.get();
            double* d_v = exp_avg_sqs[i]->cuda_data.get();
            double* d_v_max = amsgrad ? max_exp_avg_sqs[i]->cuda_data.get() : nullptr;
            launch_adam_step(
                d_param,
                d_grad,
                d_m,
                d_v,
                d_v_max,
                total_elements,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                bias_correction1,
                bias_correction2,
                maximize
            );
            continue;
        }

        // cpu-exec
        if (params[i]->grad == nullptr) continue;

        // handle non-contiguous layouts via odometer coordinate sweeps
        if (!params[i]->is_contiguous()) {
            std::vector<size_t> coords(params[i]->shape.size(), 0);
            size_t total_elements = params[i]->data->size();

            for (size_t j = 0; j < total_elements; ++j) {
                size_t flat_idx = params[i]->get_flat_index(coords);

                // extract raw gradient and adjust target execution direction orientation
                double grad = (*params[i]->grad)[flat_idx];
                if (maximize) grad = -grad;

                // inject standard regularized weight decay transformations
                if (weight_decay != 0.0) {
                    grad += weight_decay * (*params[i]->data)[flat_idx];
                }

                // update biased first and second moment tracking estimations
                (*exp_avgs[i]->data)[j] = beta1 * (*exp_avgs[i]->data)[j] + (1.0 - beta1) * grad;
                (*exp_avg_sqs[i]->data)[j] = beta2 * (*exp_avg_sqs[i]->data)[j] + (1.0 - beta2) * grad * grad;

                // compute unbiased scale expectations
                double m_hat = (*exp_avgs[i]->data)[j] / bias_correction1;
                double v_hat = 0.0;

                if (amsgrad) {
                    (*max_exp_avg_sqs[i]->data)[j] = std::max((*max_exp_avg_sqs[i]->data)[j], (*exp_avg_sqs[i]->data)[j]);
                    v_hat = (*max_exp_avg_sqs[i]->data)[j] / bias_correction2;
                } else {
                    v_hat = (*exp_avg_sqs[i]->data)[j] / bias_correction2;
                }

                // apply the calculated parameter step changes directly to host memory arrays
                (*params[i]->data)[flat_idx] -= lr * m_hat / (std::sqrt(v_hat) + eps);

                Tensor::advance_coordinates(coords, params[i]->shape);
            }
        } 
        // execute high-speed direct continuous vector array sweeps
        else {
            size_t total_elements = params[i]->data->size();
            double* param_data = params[i]->data->data();
            const double* param_grad = params[i]->grad->data();
            
            double* m_data = exp_avgs[i]->data->data();
            double* v_data = exp_avg_sqs[i]->data->data();

            if (amsgrad) {
                double* v_max_data = max_exp_avg_sqs[i]->data->data();
                
                for (size_t j = 0; j < total_elements; ++j) {
                    double grad = param_grad[j];
                    if (maximize) grad = -grad;
                    if (weight_decay != 0.0) grad += weight_decay * param_data[j];

                    m_data[j] = beta1 * m_data[j] + (1.0 - beta1) * grad;
                    v_data[j] = beta2 * v_data[j] + (1.0 - beta2) * grad * grad;

                    v_max_data[j] = std::max(v_max_data[j], v_data[j]);

                    double m_hat = m_data[j] / bias_correction1;
                    double v_hat = v_max_data[j] / bias_correction2;

                    param_data[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                }
            } else {
                for (size_t j = 0; j < total_elements; ++j) {
                    double grad = param_grad[j];
                    if (maximize) grad = -grad;
                    if (weight_decay != 0.0) grad += weight_decay * param_data[j];

                    m_data[j] = beta1 * m_data[j] + (1.0 - beta1) * grad;
                    v_data[j] = beta2 * v_data[j] + (1.0 - beta2) * grad * grad;

                    double m_hat = m_data[j] / bias_correction1;
                    double v_hat = v_data[j] / bias_correction2;

                    param_data[j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                }
            }
        }
    }
}

void Adam::zero_grad() {
    for (auto& p : params) {
        p->zero_grad();
    }
}