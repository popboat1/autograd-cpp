#include "MaxPool2D.h"
#include "utils/RNG.h"
#include <cmath>

MaxPool2D::MaxPool2D(size_t kernel_size, size_t stride)
    : kernel_size(kernel_size), stride(stride) {
}

std::vector<TensorPtr> MaxPool2D::parameters() const {
    // return an empty container since no filters exist
    return {};
}

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

    // allocate data containers and max index mask
    size_t total_out = batch_size * in_c * out_h * out_w;
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