#include "SGD.h"
#include "autograd/Tensor.h"
#include <vector>
#include "utils/cuda_utils.h"

SGD::SGD(std::vector<TensorPtr> params, double lr, double momentum, double weight_decay)
    : params(params), lr(lr), momentum_factor(momentum), wd(weight_decay) {
    
    // allocate velocity tracking arrays identical to the flat size of each parameter
    for (const auto& p : params) {
        velocities.push_back(std::make_shared<Tensor>(std::vector<double>(p->data->size(), 0.0), p->shape, false, p->device));
    }
}

__global__ void d_sgd_step(
    double* __restrict__ param,
    const double* __restrict__ grad,
    double* __restrict__ velocity,
    size_t n,
    double lr,
    double momentum,
    double weight_decay
){
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;

    // load grad
    double g = grad[idx];
    // apply regularized penalty if theres weight decay
    if(weight_decay > 0.0){
        g+= weight_decay * param[idx];
    }

    // update momentum velocity
    double v = momentum * velocity[idx] + g;
    velocity[idx] = v;
    param[idx] -= lr * v;
}

inline void launch_sgd_step(
    double* __restrict__ param,
    const double* __restrict__ grad,
    double* __restrict__ velocity,
    size_t n,
    double lr,
    double momentum,
    double weight_decay
){
    constexpr size_t block_threads = 256;
    int blocks = cuda_utils::ceil_div(static_cast<int>(n), block_threads);
    d_sgd_step<<<blocks, block_threads>>>(
        param, grad, velocity, n, lr, momentum, weight_decay
    );
    CUDA_CHECK(cudaGetLastError());
}

void SGD::step(){
    for(size_t i {0}; i < params.size(); ++i){
        // cuda device path
        if (params[i]->device == Device::CUDA) {
            // skip if no gradients were accumulated on device
            if (params[i]->cuda_grad == nullptr) continue;

            // ensure velocity tracking tensor resides on cuda
            if (velocities[i]->device != Device::CUDA) {
                velocities[i]->to(Device::CUDA);
            }

            size_t total_elements = params[i]->data->size();
            launch_sgd_step(
                params[i]->cuda_data.get(),
                params[i]->cuda_grad.get(),
                velocities[i]->cuda_data.get(),
                total_elements,
                lr,
                momentum_factor,
                wd
            );
            continue;
        }

        // cpu device path
        if (params[i]->grad == nullptr) continue;

        // handle rare non-contiguous parameters using coordinate maps
        if (!params[i]->is_contiguous()) {
            std::vector<size_t> coords(params[i]->shape.size(), 0);
            size_t total_elements = params[i]->data->size();
            
            for (size_t j = 0; j < total_elements; ++j) {
                size_t flat_idx = params[i]->get_flat_index(coords);
                
                double grad = (*params[i]->grad)[flat_idx];
                if (wd > 0.0) {
                    grad += wd * (*params[i]->data)[flat_idx];
                }
                
                // velocities[i] is guaranteed contiguous by the constructor layout allocation
                (*velocities[i]->data)[j] = (momentum_factor * (*velocities[i]->data)[j]) + grad;
                (*params[i]->data)[flat_idx] -= lr * (*velocities[i]->data)[j];
                
                Tensor::advance_coordinates(coords, params[i]->shape);
            }
        } else {
            size_t total_elements = params[i]->data->size();
            for (size_t j = 0; j < total_elements; ++j) {
                double grad = (*params[i]->grad)[j];
                if (wd > 0.0) {
                    grad += (wd * (*params[i]->data)[j]);
                }
                
                (*velocities[i]->data)[j] = (momentum_factor * (*velocities[i]->data)[j]) + grad;
                (*params[i]->data)[j] -= (lr * (*velocities[i]->data)[j]);
            }
        }
    }
}

void SGD::zero_grad() {
    for(auto& p : params){
        p->zero_grad();
    }
}