#include "AvgPool2D.h"
#include "utils/RNG.h"
#include "utils/cuda_utils.h"
#include <cmath>

AvgPool2D::AvgPool2D(size_t kernel_size, size_t stride)
    : kernel_size(kernel_size), stride(stride) {}

std::vector<TensorPtr> AvgPool2D::parameters() const {
    // return an empty container since no filters exist
    return {};
}


// -------------------------------
// CUDA kernels
// -------------------------------

__global__ void d_avgpool2d_fwd(
    const double* __restrict__ input_data,
    double* __restrict__ output_data,
    size_t total_out,
    size_t in_c, size_t in_h, size_t in_w,
    size_t out_h, size_t out_w, 
    size_t k_size, size_t stride
){
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx >= total_out) return;

    // deconstruct flat output idx into (b, ci, oh, ow)
    size_t ow = idx % out_w;
    size_t temp = idx / out_w;
    size_t oh = temp % out_h;
    temp = temp / out_h;
    size_t ci = temp % in_c;
    size_t b = temp / in_c;

    size_t in_channel_stride = in_h * in_w;
    size_t in_batch_stride = in_c * in_channel_stride;
    size_t base_src_offset = b * in_batch_stride + ci * in_channel_stride;

    double sum_val = 0.0;

    // loop over kernel window
    for(size_t kh = 0; kh < k_size; ++kh){
        size_t h_in = oh * stride + kh;
        for(size_t kw = 0; kw < k_size; ++kw){
            size_t w_in = ow * stride + kw;
            size_t src_flat_idx = base_src_offset + h_in * in_w + w_in;
            sum_val += input_data[src_flat_idx];
        }
    }

    // compute avg and write to out data
    double window_area = static_cast<double>(k_size * k_size);
    output_data[idx] = sum_val / window_area;
}

__global__ void d_avgpool2d_bwd(
    const double* __restrict__ upstream_grad,
    double* __restrict__ input_grad,
    size_t total_out,
    size_t in_c, size_t in_h, size_t in_w,    
    size_t out_h, size_t out_w,               
    size_t k_size, size_t stride
){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

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

    double scaled_grad = upstream_grad[idx] / static_cast<double>(k_size * k_size);;

    for(size_t kh = 0; kh < k_size; ++kh){
        size_t h_in = oh * stride + kh;
        for(size_t kw = 0; kw < k_size; ++kw){
            size_t w_in = ow * stride + kw;
            size_t src_flat_idx = base_src_offset + h_in * in_w + w_in;
            atomicAdd(&input_grad[src_flat_idx], scaled_grad);
        }
    }
}


// -------------------------
// CUDA launchers
// -------------------------

inline void launch_avgpool2d_forward(
    const double* d_input_data,
    double* d_output_data,
    size_t total_out,
    size_t in_c, size_t in_h, size_t in_w,
    size_t out_h, size_t out_w,
    size_t k_size, size_t stride
) {
    constexpr int block_threads = 256;
    int blocks = cuda_utils::ceil_div(static_cast<int>(total_out), block_threads);
    d_avgpool2d_fwd<<<blocks, block_threads>>>(
        d_input_data, d_output_data, total_out, 
        in_c, in_h, in_w, out_h, out_w, k_size, stride
    );
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_avgpool2d_backward(
    const double* d_upstream_grad,
    double* d_input_grad,
    size_t total_out,
    size_t in_c, size_t in_h, size_t in_w,    
    size_t out_h, size_t out_w,               
    size_t k_size, size_t stride
) {
    constexpr int block_threads = 256;
    int blocks = cuda_utils::ceil_div(static_cast<int>(total_out), block_threads);
    d_avgpool2d_bwd<<<blocks, block_threads>>>(
        d_upstream_grad, d_input_grad, total_out,
        in_c, in_h, in_w, out_h, out_w, k_size, stride
    );
    CUDA_CHECK(cudaGetLastError());
}



// ---- forward

TensorPtr AvgPool2D::forward(const TensorPtr& input) {
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

    if(x->device == Device::CUDA){
        // create output tensor for CUDA
        std::vector<double> dummy(total_out, 0.0);
        auto out = std::make_shared<Tensor>(
            std::move(dummy),
            std::vector<size_t>{batch_size, in_c, out_h, out_w},
            std::vector<TensorPtr>{x},
            "avgpool2d",
            Device::CUDA
        );
        out->requires_grad = x->requires_grad;

        // launch fwd kernel
        launch_avgpool2d_forward(x->cuda_data.get(), out->cuda_data.get(), total_out, in_c, in_h, in_w, out_h, out_w, kernel_size, stride);

        // backward pass
        std::weak_ptr<Tensor> weak_out = out;
        std::weak_ptr<Tensor> weak_x = x;
        size_t k_size = kernel_size;
        size_t s_size = stride;
        out->backward_func = [weak_x, weak_out, total_out, in_c, in_h, in_w, out_h, out_w, k_size, s_size]() {
            auto out_ptr = weak_out.lock();
            auto x_ptr = weak_x.lock();
            if (out_ptr && x_ptr && x_ptr->requires_grad) {
                x_ptr->ensure_grad_allocated();
                launch_avgpool2d_backward(
                    out_ptr->cuda_grad.get(),
                    x_ptr->cuda_grad.get(),
                    total_out,
                    in_c, in_h, in_w,
                    out_h, out_w,
                    k_size, s_size
                );
            }
        };
        return out;
    }else {
        // --- CPU
        std::vector<double> out_values(total_out, 0.0);
    
        double window_area = static_cast<double>(kernel_size * kernel_size);
    
        #pragma omp parallel for collapse(2)
        for(size_t b = 0; b < batch_size; ++b){
            for(size_t ci = 0; ci < in_c; ++ci){
                for(size_t oh = 0; oh < out_h; ++oh){
                    for(size_t ow = 0; ow < out_w; ++ow){
                        double sum_val = 0.0;
    
                        // accumulate continuous values across the sliding window footprint
                        for (size_t kh = 0; kh < kernel_size; ++kh) {
                            for (size_t kw = 0; kw < kernel_size; ++kw) {
                                size_t h_in = oh * stride + kh;
                                size_t w_in = ow * stride + kw;
    
                                size_t src_flat_idx = b * (in_c * in_h * in_w) + 
                                                      ci * (in_h * in_w) + 
                                                      h_in * in_w + 
                                                      w_in;
                                
                                sum_val += (*x->data)[src_flat_idx];
                            }
                        }
    
                        size_t dest_flat_idx = b * (in_c * out_h * out_w) + ci * (out_h * out_w) + oh * out_w + ow;
    
                        // map average calculation result into output storage array
                        out_values[dest_flat_idx] = sum_val / window_area;
                    }
                }
            }
        }
    
        auto out = std::make_shared<Tensor>(out_values, std::vector<size_t>{batch_size, in_c, out_h, out_w}, std::vector<TensorPtr>{x}, "avgpool2d");
        out->requires_grad = x->requires_grad;
        std::weak_ptr<Tensor> weak_out = out;
    
        size_t k_size = kernel_size;
        size_t s_size = stride;
    
        out->backward_func = [x, batch_size, in_c, in_h, in_w, out_h, out_w, k_size, s_size, window_area, weak_out]() {
            if (auto out_ptr = weak_out.lock()) {
                if (x->requires_grad) {
                    x->ensure_grad_allocated();
    
                    double scale_factor = 1.0 / window_area;
    
                    #pragma omp parallel for collapse(2)
                    for (size_t b = 0; b < batch_size; ++b) {
                        for (size_t ci = 0; ci < in_c; ++ci) {
                            for (size_t oh = 0; oh < out_h; ++oh) {
                                for (size_t ow = 0; ow < out_w; ++ow) {
                                    
                                    size_t dest_flat_idx = b * (in_c * out_h * out_w) + 
                                                           ci * (out_h * out_w) + 
                                                           oh * out_w + 
                                                           ow;
                                    
                                    double upstream_grad = (*out_ptr->grad)[dest_flat_idx] * scale_factor;
    
                                    for (size_t kh = 0; kh < k_size; ++kh) {
                                        for (size_t kw = 0; kw < k_size; ++kw) {
                                            size_t h_in = oh * s_size + kh;
                                            size_t w_in = ow * s_size + kw;
    
                                            size_t src_flat_idx = b * (in_c * in_h * in_w) + 
                                                                  ci * (in_h * in_w) + 
                                                                  h_in * in_w + 
                                                                  w_in;
    
                                            (*x->grad)[src_flat_idx] += upstream_grad;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
    
        return out;
    }
}