#include "Conv2D.h"
#include "utils/RNG.h"
#include <cmath>

Conv2D::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding) {
    // query the centralized global engine reference
    auto& gen = RNG::get_engine();

    // compute total flat vector capacities
    size_t weight_elements = out_channels * in_channels * kernel_size * kernel_size;
    size_t bias_elements = out_channels;

    std::vector<double> init_weights(weight_elements);
    std::vector<double> init_biases(bias_elements, 0.0);

    // he init
    double fan_in = static_cast<double>(in_channels * kernel_size * kernel_size);
    double std_dev = std::sqrt(2.0 / fan_in);

    std::normal_distribution<double> dist(0.0, std_dev);

    for(size_t i {0}; i < weight_elements; ++i){
        init_weights[i] = dist(gen);
    }
    
    // construct tensor
    weight = std::make_shared<Tensor>(init_weights, std::vector<size_t>{out_channels, in_channels, kernel_size, kernel_size}, true);
    bias = std::make_shared<Tensor>(init_biases, std::vector<size_t>{out_channels}, true);
}

std::vector<TensorPtr> Conv2D::parameters() const {
    return {weight, bias};
}

// im2col
// unroll 4d tensor patches into a row-major 2D column matrix vector
std::vector<double> Conv2D::im2col(const TensorPtr& input, size_t out_h, size_t out_w) const {
    size_t batch_size = input->shape[0];
    size_t in_c = input->shape[1];
    size_t in_h = input->shape[2];
    size_t in_w = input->shape[3];

    size_t col_rows = batch_size * out_h * out_w;
    size_t col_cols = in_channels * kernel_size * kernel_size;
    
    std::vector<double> col_matrix(col_rows * col_cols, 0.0);

    // map input patches to col_matrix rows
    for(size_t b {0}; b < batch_size; ++b){
        for(size_t oh {0}; oh < out_h; ++oh){
            for(size_t ow {0}; ow < out_w; ++ow){
                //compute the active destination row offset
                size_t curr_row = (b * out_h * out_w) + (oh * out_w) + ow;

                for(size_t ci {0}; ci < in_channels; ++ci){
                    for(size_t kh {0}; kh < kernel_size; ++kh){
                        for(size_t kw {0}; kw < kernel_size; ++kw){
                            // calculate the current column displacement within this row
                            size_t curr_col = (ci * kernel_size * kernel_size) + (kh * kernel_size) + kw;

                            // find target coordinate maps matching original image indices
                            int h_in = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                            int w_in = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

                            size_t dest_flat_idx = curr_row * col_cols + curr_col;

                            // virtual zero-padding boundary check conditional
                            if (h_in >= 0 && h_in < static_cast<int>(in_h) && 
                                w_in >= 0 && w_in < static_cast<int>(in_w)) {
                                
                                // calculate the flat contiguous index from the 4D input
                                size_t src_flat_idx = b * (in_channels * in_h * in_w) + 
                                                      ci * (in_h * in_w) + 
                                                      static_cast<size_t>(h_in) * in_w + 
                                                      static_cast<size_t>(w_in);
                                                      
                                col_matrix[dest_flat_idx] = (*input->data)[src_flat_idx];
                            } else {
                                col_matrix[dest_flat_idx] = 0.0; // padding element
                            }
                        }
                    }
                }
            }
        }
    }

    return col_matrix;
}

// col2im
// aggregate 2d column matrix values back into a 4d target image gradient array
void Conv2D::col2im(const std::vector<double>& col_grad, const TensorPtr& input_grad, size_t out_h, size_t out_w) const {
    size_t batch_size = input_grad->shape[0];
    size_t in_c = input_grad->shape[1];
    size_t in_h = input_grad->shape[2];
    size_t in_w = input_grad->shape[3];

    size_t col_cols = in_channels * kernel_size * kernel_size;

    // clean canvas for gradient accumulation
    std::fill(input_grad->grad->begin(), input_grad->grad->end(), 0.0);

    for(size_t b {0}; b < batch_size; ++b){
        for(size_t oh {0}; oh < out_h; ++oh){
            for(size_t ow {0}; ow < out_w; ++ow){
                //compute the active destination row offset
                size_t curr_row = (b * out_h * out_w) + (oh * out_w) + ow;

                for(size_t ci {0}; ci < in_channels; ++ci){
                    for(size_t kh {0}; kh < kernel_size; ++kh){
                        for(size_t kw {0}; kw < kernel_size; ++kw){
                            // calculate the current column displacement within this row
                            size_t curr_col = (ci * kernel_size * kernel_size) + (kh * kernel_size) + kw;
                            int h_in = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
                            int w_in = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

                            size_t src_flat_idx = curr_row * col_cols + curr_col;

                            // filter out padding boundaries during backward reassembly
                            if (h_in >= 0 && h_in < static_cast<int>(in_h) && 
                                w_in >= 0 && w_in < static_cast<int>(in_w)) {
                                
                                size_t dest_flat_idx = b * (in_channels * in_h * in_w) + 
                                                       ci * (in_h * in_w) + 
                                                       static_cast<size_t>(h_in) * in_w + 
                                                       static_cast<size_t>(w_in);
                                                       
                                // accumulate gradients into matching coordinates
                                (*input_grad->grad)[dest_flat_idx] += col_grad[src_flat_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

TensorPtr Conv2D::forward(const TensorPtr& input){
    auto x = input->is_contiguous() ? input : input->contiguous();

    // unpack dimensions
    size_t batch_size = x->shape[0];
    size_t in_c = x->shape[1];
    size_t in_h = x->shape[2];
    size_t in_w = x->shape[3];

    // calculate output spatial boundary maps
    size_t out_h = ((in_h - kernel_size + 2 * padding) / stride) + 1;
    size_t out_w = ((in_w - kernel_size + 2 * padding) / stride) + 1;

    std::vector<double> col_data = im2col(x, out_h, out_w);

    // package unrolled parameters into a tracking graph node
    size_t col_rows = batch_size * out_h * out_w;
    size_t col_cols = in_channels * kernel_size * kernel_size;
    auto input_col_tensor = std::make_shared<Tensor>(col_data, std::vector<size_t>{col_rows, col_cols}, std::vector<TensorPtr>{x}, "im2col");

    // flatten weights parameters to 2D footprint using existing tool
    auto weights_2d = weight->view({static_cast<int>(out_channels), static_cast<int>(col_cols)});

    // general matrix multiplication (gemm) pass invocation
    auto weights_t = weights_2d->transpose(0, 1);
    auto gemm_out = Tensor::matmul(input_col_tensor, weights_t);

    // accumulate layer channel biases via standard broadcasting
    auto gemm_out_biased = gemm_out + bias;
    
    // reconstruct back into framework standard 4D vision shape
    auto final_4d = gemm_out_biased->view({
        static_cast<int>(batch_size), 
        static_cast<int>(out_channels), 
        static_cast<int>(out_h), 
        static_cast<int>(out_w)
    });

    input_col_tensor->backward_func = [this, x, input_col_tensor, out_h, out_w]() {
        if (x->requires_grad) {
            this->col2im(*input_col_tensor->grad, x, out_h, out_w);
        }
    };

    return final_4d;
}