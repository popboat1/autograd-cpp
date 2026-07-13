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

    for(size_t i = 0; i < weight_elements; ++i){
        init_weights[i] = dist(gen);
    }
    
    // construct tensor
    weight = std::make_shared<Tensor>(std::move(init_weights), std::vector<size_t>{out_channels, in_channels, kernel_size, kernel_size}, true);
    bias = std::make_shared<Tensor>(std::move(init_biases), std::vector<size_t>{out_channels}, true);
}

std::vector<TensorPtr> Conv2D::parameters() const {
    return {weight, bias};
}

// im2col
// unroll 4d tensor patches into a row-major 2D column matrix vector
// parallelized using linear pointer tracking and branch-hoisting padding
std::vector<double> Conv2D::im2col(const TensorPtr& input, size_t out_h, size_t out_w) const {
    size_t batch_size = input->shape[0];
    size_t in_h = input->shape[2];
    size_t in_w = input->shape[3];

    size_t col_rows = batch_size * out_h * out_w;
    size_t col_cols = in_channels * kernel_size * kernel_size;
    
    std::vector<double> col_matrix(col_rows * col_cols);
    double* col_ptr = col_matrix.data();
    const double* input_ptr = input->data->data();

    size_t img_stride = in_channels * in_h * in_w;
    size_t ch_stride = in_h * in_w;

    // map input patches to col_matrix rows
    // map input patches to col_matrix rows
    #pragma omp parallel for collapse(3) schedule(static)
    for(size_t b = 0; b < batch_size; ++b){
        for(size_t oh = 0; oh < out_h; ++oh){
            for(size_t ow = 0; ow < out_w; ++ow){
                // compute the active destination row offset
                size_t curr_row = (b * out_h * out_w) + (oh * out_w) + ow;
                double* local_col_ptr = col_ptr + (curr_row * col_cols);
                size_t write_idx = 0;

                for(size_t ci = 0; ci < in_channels; ++ci){
                    size_t src_channel_offset = b * img_stride + ci * ch_stride;

                    for(size_t kh = 0; kh < kernel_size; ++kh){
                        int h_in = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);

                        // hoist height check outside the innermost loop to prevent branch mispredictions
                        if (h_in >= 0 && h_in < static_cast<int>(in_h)) {
                            size_t src_row_offset = src_channel_offset + static_cast<size_t>(h_in) * in_w;
                            
                            for(size_t kw = 0; kw < kernel_size; ++kw){
                                int w_in = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);
                                if (w_in >= 0 && w_in < static_cast<int>(in_w)) {
                                    local_col_ptr[write_idx++] = input_ptr[src_row_offset + static_cast<size_t>(w_in)];
                                } else {
                                    local_col_ptr[write_idx++] = 0.0;
                                }
                            }
                        } else {
                            // fill complete padded rows with zero instantly
                            for(size_t kw = 0; kw < kernel_size; ++kw){
                                local_col_ptr[write_idx++] = 0.0;
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
    size_t in_h = input_grad->shape[2];
    size_t in_w = input_grad->shape[3];

    size_t col_cols = in_channels * kernel_size * kernel_size;

    std::fill(input_grad->grad->begin(), input_grad->grad->end(), 0.0);
    double* grad_ptr = input_grad->grad->data();
    const double* col_grad_ptr = col_grad.data();

    size_t img_stride = in_channels * in_h * in_w;
    size_t ch_stride = in_h * in_w;

    #pragma omp parallel for collapse(2) schedule(static)
    for(size_t b = 0; b < batch_size; ++b){
        for(size_t ci = 0; ci < in_channels; ++ci){
            size_t dest_channel_offset = b * img_stride + ci * ch_stride;

            for(size_t oh = 0; oh < out_h; ++oh){
                for(size_t ow = 0; ow < out_w; ++ow){
                    // compute the active destination row offset
                    size_t curr_row = (b * out_h * out_w) + (oh * out_w) + ow;

                    const double* local_col_grad_row = col_grad_ptr + (curr_row * col_cols);
                    size_t col_channel_offset = ci * kernel_size * kernel_size;

                    for(size_t kh = 0; kh < kernel_size; ++kh){
                        int h_in = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);

                        if (h_in >= 0 && h_in < static_cast<int>(in_h)) {
                            size_t dest_row_offset = dest_channel_offset + static_cast<size_t>(h_in) * in_w;
                            size_t col_row_offset = col_channel_offset + kh * kernel_size;

                            for(size_t kw = 0; kw < kernel_size; ++kw){
                                int w_in = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);
                                if (w_in >= 0 && w_in < static_cast<int>(in_w)) {
                                    grad_ptr[dest_row_offset + static_cast<size_t>(w_in)] += 
                                        local_col_grad_row[col_row_offset + kw];
                                }
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
    auto input_col_tensor = std::make_shared<Tensor>(std::move(col_data), std::vector<size_t>{col_rows, col_cols}, std::vector<TensorPtr>{x}, "im2col");

    // flatten weights parameters to 2D footprint using existing tool
    auto weights_2d = weight->view({static_cast<int>(out_channels), static_cast<int>(col_cols)});

    // general matrix multiplication (gemm) pass invocation
    auto weights_t = weights_2d->transpose(0, 1);
    auto gemm_out = Tensor::matmul(input_col_tensor, weights_t);

    // accumulate layer channel biases via standard broadcasting
    auto gemm_out_biased = gemm_out + bias;

    // map the true NHWC layout first
    auto intermediate_nhwc = gemm_out_biased->view({
        static_cast<int>(batch_size), 
        static_cast<int>(out_h), 
        static_cast<int>(out_w),
        static_cast<int>(out_channels)
    });
    
    // permute axes safely from NHWC [0, 1, 2, 3] to NCHW [0, 3, 1, 2]
    auto final_4d = intermediate_nhwc->permute({0, 3, 1, 2});

    // establish safe weak references to break cyclic tracking loops and avoid dangling pointers
    std::weak_ptr<Tensor> weak_col = input_col_tensor;
    std::weak_ptr<Tensor> weak_x = x;
    std::weak_ptr<const Conv2D> weak_this = shared_from_this();

    input_col_tensor->backward_func = [weak_this, weak_x, weak_col, out_h, out_w]() {
        auto x_ptr = weak_x.lock();
        auto col_ptr = weak_col.lock();
        auto this_ptr = weak_this.lock();
        
        // only execute backpropagation if all components are alive in memory
        if (this_ptr && x_ptr && col_ptr) {
            if (x_ptr->requires_grad) {
                this_ptr->col2im(*col_ptr->grad, x_ptr, out_h, out_w);
            }
        }
    };

    return final_4d;
}