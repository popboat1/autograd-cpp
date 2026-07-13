#include "AvgPool2D.h"
#include "utils/RNG.h"
#include <cmath>

AvgPool2D::AvgPool2D(size_t kernel_size, size_t stride)
    : kernel_size(kernel_size), stride(stride) {}

std::vector<TensorPtr> AvgPool2D::parameters() const {
    // return an empty container since no filters exist
    return {};
}

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

    // allocate data containers and max index mask
    size_t total_out = batch_size * in_c * out_h * out_w;
    std::vector<double> out_values(total_out, 0.0);

    double window_area = static_cast<double>(kernel_size * kernel_size);

    #pragma omp parallel for collapse(2)
    for(size_t b {0}; b < batch_size; ++b){
        for(size_t ci {0}; ci < in_c; ++ci){
            for(size_t oh {0}; oh < out_h; ++oh){
                for(size_t ow {0}; ow < out_w; ++ow){
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