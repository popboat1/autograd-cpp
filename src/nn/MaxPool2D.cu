#include "MaxPool2D.h"
#include "utils/RNG.h"
#include "utils/cuda_utils.h"
#include <cmath>

MaxPool2D::MaxPool2D(size_t kernel_size, size_t stride)
    : kernel_size(kernel_size), stride(stride) {
}

std::vector<TensorPtr> MaxPool2D::parameters() const {
    // return an empty container since no filters exist
    return {};
}

// ---------------------------------------------------------------------------
// CUDA Kernels for MaxPool2D
// ---------------------------------------------------------------------------

__global__ void d_maxpool2d_fwd(
    const double* __restrict__ input_data,
    double* __restrict__ output_data,
    size_t* __restrict__ max_indices,
    size_t total_out,
    size_t in_c, size_t in_h, size_t in_w,
    size_t out_h, size_t out_w,
    size_t k_size, size_t stride
) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= total_out) return;

    // deconstruct flat output index into (b, ci, oh, ow)
    size_t ow = idx % out_w;
    size_t temp = idx / out_w;
    size_t oh = temp % out_h;
    temp = temp / out_h;
    size_t ci = temp % in_c;
    size_t b = temp / in_c;

    size_t in_channel_stride = in_h * in_w;
    size_t in_batch_stride = in_c * in_channel_stride;
    size_t base_src_offset = b * in_batch_stride + ci * in_channel_stride;

    double max_val = -INFINITY;
    size_t winning_src_idx = 0;

    for (size_t kh = 0; kh < k_size; ++kh){
        size_t h_in = oh * stride + kh;
        for(size_t kw = 0; kw < k_size; ++kw){
            size_t w_in = ow * stride + kw;
            size_t src_flat_idx = base_src_offset + h_in * in_w + w_in;
            double val = input_data[src_flat_idx];
            if (val > max_val) {
                max_val = val;
                winning_src_idx = src_flat_idx;
            }
        }
    }

    output_data[idx] = max_val;
    max_indices[idx] = winning_src_idx;
}

__global__ void d_maxpool2d_bckwrd(
    const double* __restrict__ upstream_grad,
    double* __restrict__ input_grad,
    const size_t* __restrict__ max_indices,
    size_t total_out
){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    size_t input_winner_idx = max_indices[idx];
    atomicAdd(&input_grad[input_winner_idx], upstream_grad[idx]);
}

// --------------------------------
// CUDA launchers
// --------------------------------

inline void launch_maxpool2d_forward(
    const double* d_input_data,
    double* d_output_data,
    size_t* d_max_indices,
    size_t total_out,
    size_t in_c, size_t in_h, size_t in_w,
    size_t out_h, size_t out_w,
    size_t k_size, size_t stride
) {
    constexpr int block_threads = 256;
    int blocks = cuda_utils::ceil_div(static_cast<int>(total_out), block_threads);
    d_maxpool2d_fwd<<<blocks, block_threads>>>(
        d_input_data, d_output_data, d_max_indices,
        total_out, in_c, in_h, in_w, out_h, out_w, k_size, stride
    );
    CUDA_CHECK(cudaGetLastError());
}
inline void launch_maxpool2d_backward(
    const double* d_upstream_grad,
    double* d_input_grad,
    const size_t* d_max_indices,
    size_t total_out
) {
    constexpr int block_threads = 256;
    int blocks = cuda_utils::ceil_div(static_cast<int>(total_out), block_threads);
    d_maxpool2d_bckwrd<<<blocks, block_threads>>>(
        d_upstream_grad, d_input_grad, d_max_indices, total_out
    );
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// MaxPool2D::forward
// ---------------------------------------------------------------------------

TensorPtr MaxPool2D::forward(const TensorPtr& input) {
    auto x = input->is_contiguous() ? input : input->contiguous();

    // unpack dims
    size_t batch_size = x->shape[0];
    size_t in_c = x->shape[1];
    size_t in_h = x->shape[2];
    size_t in_w = x->shape[3];

    // calculate output map boundaries
    size_t out_h = ((in_h - kernel_size) / stride) + 1;
    size_t out_w = ((in_w - kernel_size) / stride) + 1;
    size_t total_out = batch_size * in_c * out_h * out_w;

    if (x->device == Device::CUDA) {
        std::vector<double> dummy(total_out, 0.0);
        auto out = std::make_shared<Tensor>(
            std::move(dummy),
            std::vector<size_t>{batch_size, in_c, out_h, out_w},
            std::vector<TensorPtr>{x},
            "maxpool2d",
            Device::CUDA
        );
        out->requires_grad = x->requires_grad;

        // allocate device memory for max indices with automatic RAII cleanup
        size_t* raw_indices = nullptr;
        CUDA_CHECK(cudaMalloc(&raw_indices, total_out * sizeof(size_t)));
        auto d_max_indices = std::shared_ptr<size_t>(raw_indices, [](size_t* ptr) {
            if (ptr) cudaFree(ptr);
        });

        launch_maxpool2d_forward(
            x->cuda_data.get(),
            out->cuda_data.get(),
            d_max_indices.get(),
            total_out,
            in_c, in_h, in_w,
            out_h, out_w,
            kernel_size, stride
        );


        // backward pass
        std::weak_ptr<Tensor> weak_out = out;
        std::weak_ptr<Tensor> weak_x = x;
        out->backward_func = [weak_x, weak_out, d_max_indices, total_out]() {
            auto out_ptr = weak_out.lock();
            auto x_ptr = weak_x.lock();
            if (out_ptr && x_ptr && x_ptr->requires_grad) {
                x_ptr->ensure_grad_allocated();
                launch_maxpool2d_backward(
                    out_ptr->cuda_grad.get(),
                    x_ptr->cuda_grad.get(),
                    d_max_indices.get(),
                    total_out
                );
            }
        };

        return out;
    } else {
        // --- CPU
        std::vector<double> out_values(total_out, 0.0);
        auto max_indices = std::make_shared<std::vector<size_t>>(total_out, 0);
    
        for(size_t b {0}; b < batch_size; ++b){
            for(size_t ci {0}; ci < in_c; ++ci){
                for(size_t oh {0}; oh < out_h; ++oh){
                    for(size_t ow {0}; ow < out_w; ++ow){
                        double max_val = -INFINITY;
                        size_t winning_flat_idx = 0;
    
                        for(size_t kh {0}; kh < kernel_size; ++kh){
                            for(size_t kw {0}; kw < kernel_size; ++kw){
                                size_t h_in = oh * stride + kh;
                                size_t w_in = ow * stride + kw;
    
                                size_t src_flat_idx = b * (in_c * in_h * in_w) + ci * (in_h * in_w) + h_in * in_w + w_in;
    
                                if((*x->data)[src_flat_idx] > max_val){
                                    max_val = (*x->data)[src_flat_idx];
                                    winning_flat_idx = src_flat_idx;
                                }
                            }
                        }
    
                        size_t dest_flat_idx = b * (in_c * out_h * out_w) + ci * (out_h * out_w) + oh * out_w + ow;
    
                        out_values[dest_flat_idx] = max_val;
                        (*max_indices)[dest_flat_idx] = winning_flat_idx;
                    }
                }
            }
        }
    
        auto out = std::make_shared<Tensor>(out_values, std::vector<size_t>{batch_size, in_c, out_h, out_w}, std::vector<TensorPtr>{x}, "maxpool2d");
        out->requires_grad = x->requires_grad;
        std::weak_ptr<Tensor> weak_out = out;
        out->backward_func = [x, max_indices, total_out, weak_out](){
            if(auto out_ptr = weak_out.lock()){
                if (x->requires_grad){
                    x->ensure_grad_allocated();
    
                    for(size_t i {0}; i < total_out; ++i){
                        // pull out the flat winning index address from max_indices
                        size_t input_winner_idx = (*max_indices)[i];
    
                        // route the upstream gradient value back into x->grad at that address
                        (*x->grad)[input_winner_idx] += (*out_ptr->grad)[i];
                    }
                }
            }
        };
        return out;

    }
}