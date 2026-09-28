#include "BatchNorm2D.h"
#include <cmath>
#include <stdexcept>
#include "utils/cuda_utils.h"

BatchNorm2D::BatchNorm2D(size_t num_features, double eps, double momentum, 
                         bool affine, bool track_running_stats)
    : num_features(num_features), eps(eps), momentum(momentum), 
      affine(affine), track_running_stats(track_running_stats), training(true) {

    std::vector<size_t> param_shape = {1, num_features, 1, 1};

    if(affine){
        // gamma init to 1.0, beta to 0.0
        weight = std::make_shared<Tensor>(std::vector<double>(num_features, 1.0), param_shape, true);
        bias = std::make_shared<Tensor>(std::vector<double>(num_features, 0.0), param_shape, true);
    }

    if(track_running_stats){
        running_mean = std::make_shared<Tensor>(std::vector<double>(num_features, 0.0), param_shape, false);
        running_var = std::make_shared<Tensor>(std::vector<double>(num_features, 1.0), param_shape, false);
    }
}

std::vector<TensorPtr> BatchNorm2D::parameters() const{
    std::vector<TensorPtr> params;
    if(affine){
        params.push_back(weight);
        params.push_back(bias);
    }
    return params;
}

// CUDA kernels for BatchNorm's running stats
__global__ void d_batchnorm2d_update_stats(
    double* __restrict__ running_mean,
    double* __restrict__ running_var,
    const double* __restrict__ current_mean,
    const double* __restrict__ current_var,
    size_t num_features,
    double momentum, 
    double unbiased_var_scale
) {
    size_t c = threadIdx.x + blockDim.x * blockIdx.x;
    if(c >= num_features) return;

    double mean_val = current_mean[c];
    double unbiased_var = current_var[c] * unbiased_var_scale;

    running_mean[c] = (1.0 - momentum) * running_mean[c] + momentum * mean_val;
    running_var[c] = (1.0 - momentum) * running_var[c] + momentum * unbiased_var;
}

inline void launch_batchnorm2d_update_stats(
    double* d_running_mean,
    double* d_running_var,
    const double* d_current_mean,
    const double* d_current_var,
    size_t num_features,
    double momentum,
    double unbiased_var_scale
) {
    constexpr int block_threads = 256;
    int blocks = cuda_utils::ceil_div(static_cast<int>(num_features), block_threads);
    d_batchnorm2d_update_stats<<<blocks, block_threads>>>(
        d_running_mean, d_running_var, d_current_mean, d_current_var,
        num_features, momentum, unbiased_var_scale
    );
    CUDA_CHECK(cudaGetLastError());
}

TensorPtr BatchNorm2D::forward(const TensorPtr& input) {
    if (input->shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d input must be a 4D tensor matching NCHW geometry layout");
    }

    size_t batch_size = input->shape[0];
    size_t channels = input->shape[1];
    size_t height = input->shape[2];
    size_t width = input->shape[3];

    if (channels != num_features) {
        throw std::invalid_argument("input channels dimension mismatch with batchnorm structural features initialization");
    }

    if (affine) {
        if (weight->device != input->device) weight->to(input->device);
        if (bias->device != input->device) bias->to(input->device);
    }
    if (track_running_stats) {
        if (running_mean->device != input->device) running_mean->to(input->device);
        if (running_var->device != input->device) running_var->to(input->device);
    }

    auto eps_tensor = std::make_shared<Tensor>(
        std::vector<double>{eps}, std::vector<size_t>{1, 1, 1, 1}, false, input->device
    );
    TensorPtr x_hat;

    // check module's runtime training flag inherited from Module base
    if (this->training) {
        // calculate channel-wide mean values by reducing W, H, then B
        auto mean_w = input->mean(3, true);
        auto mean_wh = mean_w->mean(2, true);
        auto mean_batch = mean_wh->mean(0, true); // final shape: [1, C, 1, 1]

        // calculate variance using broadcasted difference squares
        auto diff = input - mean_batch;
        auto sq_diff = diff * diff;
        auto var_w = sq_diff->mean(3, true);
        auto var_wh = var_w->mean(2, true);
        auto var_batch = var_wh->mean(0, true); // final shape: [1, C, 1, 1]

        // calculate continuous normalized tracking outputs using the epsilon tensor node
        auto denom = (var_batch + eps_tensor)->sqrt();
        x_hat = diff / denom;

        // non-differentiable moving average updates
        if (track_running_stats) {
            double N = static_cast<double>(batch_size * height * width);
            double unbiased_var_scale = (N > 1.0) ? (N / (N - 1.0)) : 1.0;

            if (input->device == Device::CUDA) {
                launch_batchnorm2d_update_stats(
                    running_mean->cuda_data.get(),
                    running_var->cuda_data.get(),
                    mean_batch->cuda_data.get(),
                    var_batch->cuda_data.get(),
                    num_features,
                    momentum,
                    unbiased_var_scale
                );
            } else {
                for (size_t c = 0; c < num_features; ++c) {
                    double current_mean = mean_batch->data->data()[c];
                    double current_biased_var = var_batch->data->data()[c];
                    double current_unbiased_var = current_biased_var * unbiased_var_scale;

                    // apply exponential moving average updates to internal arrays
                    running_mean->data->data()[c] = (1.0 - momentum) * running_mean->data->data()[c] + momentum * current_mean;
                    running_var->data->data()[c] = (1.0 - momentum) * running_var->data->data()[c] + momentum * current_unbiased_var;
                }
            }
        }
    } 
    // execute static inference pass using running statistics
    else {
        if (!track_running_stats) {
            throw std::runtime_error("cannot run evaluation pass without active tracked running statistics states");
        }
        auto diff = input - running_mean;
        auto denom = (running_var + eps_tensor)->sqrt();
        x_hat = diff / denom;
    }

    // apply scale (gamma) and shift (beta) transformations via broadcasting
    if (affine) {
        return x_hat * weight + bias;
    }
    return x_hat;
}