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
std::shared_ptr<std::vector<double>> Conv2D::im2col(const TensorPtr& input, size_t out_h, size_t out_w) const {
    size_t batch_size = input->shape[0];
    size_t in_h = input->shape[2];
    size_t in_w = input->shape[3];

    size_t col_rows = batch_size * out_h * out_w;
    size_t col_cols = in_channels * kernel_size * kernel_size;
    size_t required_size = col_rows * col_cols;
    
    if (!col_matrix) {
        col_matrix = std::make_shared<std::vector<double>>(required_size);
    } else if (col_matrix->size() != required_size) {
        col_matrix->resize(required_size);
    }
    
    double* col_ptr = col_matrix->data();
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

// im2col kernel
__global__ void d_im2col(
    const double* __restrict__ input_data,
    double* __restrict__ col_data,
    size_t total_elements,
    size_t batch_size,
    size_t in_c, size_t in_h, size_t in_w,
    size_t out_h, size_t out_w,
    size_t k_size, size_t stride, size_t padding
){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    size_t col_cols = in_c * k_size * k_size;
    size_t spatial_out = out_h * out_w;

    // deconstruct flat idx into matrix (row, col)
    size_t row = idx / col_cols;
    size_t col = idx % col_cols;

    // deconstruct matrix into (b, oh, ow)
    size_t b = row / spatial_out;
    size_t spatial_idx = row % spatial_out;
    size_t oh = spatial_idx / out_w;
    size_t ow = spatial_idx % out_w;

    // deconstruct matrix cols into (ci, kh, kw)
    size_t kernel_spatial = k_size * k_size;
    size_t ci = col / kernel_spatial;
    size_t k_idx = col % kernel_spatial;
    size_t kh = k_idx / k_size;
    size_t kw = k_idx % k_size;

    // map to input coordinates (with stride and padding)
    int h_in = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
    int w_in = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

    // zero-padding check and coalesced write
    if (h_in >= 0 && h_in < static_cast<int>(in_h) &&
        w_in >= 0 && w_in < static_cast<int>(in_w)) {
        size_t input_idx = b * (in_c * in_h * in_w) +
                           ci * (in_h * in_w) +
                           static_cast<size_t>(h_in) * in_w +
                           static_cast<size_t>(w_in);
        col_data[idx] = input_data[input_idx];
    } else {
        col_data[idx] = 0.0;
    }
}

// im2col kernel launcher
inline void launch_im2col_forward(
    const double* d_input,
    double* d_col,
    size_t batch_size,
    size_t in_c, size_t in_h, size_t in_w,
    size_t out_h, size_t out_w,
    size_t k_size, size_t stride, size_t padding
) {
    size_t total_elements = (batch_size * out_h * out_w) * (in_c * k_size * k_size);
    constexpr int block_threads = 256;
    int blocks = cuda_utils::ceil_div(static_cast<int>(total_elements), block_threads);

    d_im2col<<<blocks, block_threads>>>(
        d_input, d_col, total_elements,
        batch_size, in_c, in_h, in_w,
        out_h, out_w, k_size, stride, padding
    );
    CUDA_CHECK(cudaGetLastError());
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

// col2im kernel
__global__ void d_col2im(
    const double* __restrict__ col_grad,
    double* __restrict__ input_grad,
    size_t total_elements,
    size_t batch_size,
    size_t in_c, size_t in_h, size_t in_w,
    size_t out_h, size_t out_w,
    size_t k_size, size_t stride, size_t padding
){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_elements) return;

    size_t col_cols = in_c * k_size * k_size;
    size_t spatial_out = out_w * out_h;

    // deconstruct flat idx into matrix (row, col)
    size_t row = idx / col_cols;
    size_t col = idx % col_cols;

    // deconstruct matrix row into (b, oh, ow)
    size_t b = row / spatial_out;
    size_t spatial_idx = row % spatial_out;
    size_t oh = spatial_idx / out_w;
    size_t ow = spatial_idx % out_w;

    // deconstruct matrix col into (ci, kh, kw)
    size_t kernel_spatial = k_size * k_size;
    size_t ci = col / kernel_spatial;
    size_t k_idx = col % kernel_spatial;
    size_t kh = k_idx / k_size;
    size_t kw = k_idx % k_size;

    // map to original input image coordinates
    int h_in = static_cast<int>(oh * stride + kh) - static_cast<int>(padding);
    int w_in = static_cast<int>(ow * stride + kw) - static_cast<int>(padding);

    // scatter-accumulate gradients back into input tensor
    if (h_in >= 0 && h_in < static_cast<int>(in_h) &&
        w_in >= 0 && w_in < static_cast<int>(in_w)) {
        size_t input_idx = b * (in_c * in_h * in_w) +
                           ci * (in_h * in_w) +
                           static_cast<size_t>(h_in) * in_w +
                           static_cast<size_t>(w_in);
        atomicAdd(&input_grad[input_idx], col_grad[idx]);
    }
}

// col2im launcher
inline void launch_col2im_backward(
    const double* d_col_grad,
    double* d_input_grad,
    size_t batch_size,
    size_t in_c, size_t in_h, size_t in_w,
    size_t out_h, size_t out_w,
    size_t k_size, size_t stride, size_t padding
) {
    size_t total_elements = (batch_size * out_h * out_w) * (in_c * k_size * k_size);
    constexpr int block_threads = 256;
    int blocks = cuda_utils::ceil_div(static_cast<int>(total_elements), block_threads);

    d_col2im<<<blocks, block_threads>>>(
        d_col_grad, d_input_grad, total_elements,
        batch_size, in_c, in_h, in_w,
        out_h, out_w, k_size, stride, padding
    );
    CUDA_CHECK(cudaGetLastError());
}

TensorPtr Conv2D::forward(const TensorPtr& input){
    auto x = input->is_contiguous() ? input : input->contiguous();

    // synchronize parameter device placement
    if (weight->device != x->device) weight->to(x->device);
    if (bias->device != x->device) bias->to(x->device);

    // unpack dimensions
    size_t batch_size = x->shape[0];
    size_t in_c = x->shape[1];
    size_t in_h = x->shape[2];
    size_t in_w = x->shape[3];

    // calculate output spatial boundary maps
    size_t out_h = ((in_h - kernel_size + 2 * padding) / stride) + 1;
    size_t out_w = ((in_w - kernel_size + 2 * padding) / stride) + 1;

    // package unrolled parameters into a tracking graph node
    size_t col_rows = batch_size * out_h * out_w;
    size_t col_cols = in_channels * kernel_size * kernel_size;

    TensorPtr input_col_tensor = nullptr;

    // forward im2col
    if(x->device == Device::CUDA){
        std::vector<double> dummy(col_rows * col_cols, 0.0);
        input_col_tensor = std::make_shared<Tensor>(
            std::move(dummy),
            std::vector<size_t>{col_rows, col_cols},
            std::vector<TensorPtr>{x},
            "im2col",
            Device::CUDA
        );

        launch_im2col_forward(
            x->cuda_data.get(),
            input_col_tensor->cuda_data.get(),
            batch_size, in_c, in_h, in_w,
            out_h, out_w, kernel_size, stride, padding
        );
    } else {
        auto col_data_ptr = im2col(x, out_h, out_w);
        input_col_tensor = std::make_shared<Tensor>(
            col_data_ptr,
            nullptr,
            std::vector<size_t>{col_rows, col_cols},
            std::vector<TensorPtr>{x},
            "im2col",
            Device::CPU
        );
    }

    // flatten weights parameters to 2D footprint using existing tool
    auto weights_2d = weight->view({static_cast<int>(out_channels), static_cast<int>(col_cols)});

    // general matrix multiplication (gemm) pass invocation
    auto weights_t = weights_2d->transpose(0, 1)->contiguous();
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
        if (this_ptr && x_ptr && col_ptr && x_ptr->requires_grad) {
            if (x_ptr->device == Device::CUDA) {
                launch_col2im_backward(
                    col_ptr->cuda_grad.get(),
                    x_ptr->cuda_grad.get(),
                    x_ptr->shape[0], x_ptr->shape[1], x_ptr->shape[2], x_ptr->shape[3],
                    out_h, out_w,
                    this_ptr->kernel_size, this_ptr->stride, this_ptr->padding
                );
            } else {
                this_ptr->col2im(*col_ptr->grad, x_ptr, out_h, out_w);
            }
        }
    };

    return final_4d;
}