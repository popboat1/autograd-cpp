#include "Tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>


// --------------------------------
// constructors
// --------------------------------


// constructor for leaf nodes / primary constructor
Tensor::Tensor(std::vector<double> values, std::vector<size_t> shape, bool requires_grad, Device device)
    : data(std::make_shared<std::vector<double>>(std::move(values))),
      grad(nullptr),
      shape(std::move(shape)), requires_grad(requires_grad), op(""), backward_func([](){}), device(device) {
    
    // compute row-major strides
    strides.resize(this->shape.size(), 1);
    if (!this->shape.empty()){
        for(int i {static_cast<int>(this->shape.size()) - 2}; i >= 0; --i){
            strides[i] = strides[i + 1] * this->shape[i + 1];
        }
    }
}

// graph constructor for operations
Tensor::Tensor(std::vector<double> values, std::vector<size_t> shape, std::vector<TensorPtr> children, std::string operation, Device device)
    : data(std::make_shared<std::vector<double>>(std::move(values))), 
      grad(nullptr),
      shape(std::move(shape)), requires_grad(false), prev(std::move(children)), op(std::move(operation)), backward_func([](){}), device(device){

    // compute strides for the new shape layout
    strides.resize(this->shape.size(), 1);
    if (!this->shape.empty()) {
        for (int i = static_cast<int>(this->shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * this->shape[i + 1];
        }
    }

    // inherit tracking state from unique parent nodes
    for (const auto& child : prev) {
        if (child->requires_grad) {
            this->requires_grad = true;
        }
    }
}

// zero-copy view constructor
Tensor::Tensor(std::shared_ptr<std::vector<double>> shared_data, std::shared_ptr<std::vector<double>> shared_grad, std::vector<size_t> shape, std::vector<TensorPtr> children, std::string operation, Device device)
    : data(std::move(shared_data)), grad(std::move(shared_grad)), shape(std::move(shape)), requires_grad(false), prev(std::move(children)), op(std::move(operation)), backward_func([](){}), device(device) {
    
    strides.resize(this->shape.size(), 1);
    if (!this->shape.empty()) {
        for (int i = static_cast<int>(this->shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * this->shape[i + 1];
        }
    }

    for (const auto& child : prev) {
        if (child->requires_grad) this->requires_grad = true;
    }
}

// end of constructors....
// --------------------------------

// device function to move tensors between cuda and cpu
void Tensor::to(Device target_device){
    if (this->device == target_device) return;

    size_t bytes = data->size() * sizeof(double);

    if(target_device == Device::CUDA){
        // cpu -> gpu
        CUDA_CHECK(cudaMalloc(&cuda_data, bytes));
        CUDA_CHECK(cudaMemcpy(cuda_data, data->data(), bytes, cudaMemcpyHostToDevice));

        if(grad){
            CUDA_CHECK(cudaMalloc(&cuda_grad, bytes));
            CUDA_CHECK(cudaMemcpy(cuda_grad, grad->data(), bytes, cudaMemcpyHostToDevice));
        }
    } else if(target_device == Device::CPU){
        // gpu -> cpu
        CUDA_CHECK(cudaMemcpy(data->data(), cuda_data, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(cuda_data));
        cuda_data = nullptr;

        if(grad && cuda_grad){
            CUDA_CHECK(cudaMemcpy(grad->data(), cuda_grad, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(cuda_grad));
            cuda_grad = nullptr;
        }
    }

    this->device = target_device;
}

// destructor
Tensor::~Tensor(){
    if (cuda_data) {
        if (cudaFree(cuda_data) != cudaSuccess) {
            std::cerr << "fatal error: Failed to free cuda data!" << std::endl;
        }
    }
    if (cuda_grad) {
        if (cudaFree(cuda_grad) != cudaSuccess) {
            std::cerr << "fatal error: Failed to free cuda grad!" << std::endl;
        }
    }
}



// index mapping
size_t Tensor::get_flat_index(const std::vector<size_t>& indices) const {
    if(indices.size() != shape.size()){
        throw std::invalid_argument("tensor index mismatch dimension");
    }

    size_t flat_idx = 0;
    for (size_t i {0}; i < indices.size(); ++i){    
        if(indices[i] >= shape[i]){
            throw std::out_of_range("tensor index out of bounds");
        }
        flat_idx += indices[i] * strides[i];
    }
    return flat_idx;
}

// computes broadcasted shapes
void Tensor::compute_broadcast_metadata(
    const TensorPtr& lhs, const TensorPtr& rhs,
    std::vector<size_t>& out_shape,
    std::vector<size_t>& lhs_b_strides,
    std::vector<size_t>& rhs_b_strides
){
    size_t out_ndim = std::max(lhs->shape.size(), rhs->shape.size());

    out_shape.resize(out_ndim);
    lhs_b_strides.resize(out_ndim);
    rhs_b_strides.resize(out_ndim);

    for(size_t i {0}; i < out_ndim; ++i){
        size_t out_idx = out_ndim - 1 - i;

        // extract size and strides for lhs
        size_t dim_lhs = 1;
        size_t stride_lhs = 0;
        if (i < lhs->shape.size()){
            dim_lhs = lhs->shape[lhs->shape.size() - 1 - i];
            stride_lhs = lhs->strides[lhs->shape.size() - 1 - i];
        }

        // extract size and strides for rhs
        size_t dim_rhs = 1;
        size_t stride_rhs = 0;
        if(i < rhs->shape.size()){
            dim_rhs = rhs->shape[rhs->shape.size() - 1 - i];
            stride_rhs = rhs->strides[rhs->shape.size() - 1 - i];
        }

        // broadcasting rule comparison matrix
        if (dim_lhs == dim_rhs){
            out_shape[out_idx] = dim_lhs;
            lhs_b_strides[out_idx] = stride_lhs;
            rhs_b_strides[out_idx] = stride_rhs;
        } else if (dim_lhs == 1) {
            out_shape[out_idx] = dim_rhs;
            lhs_b_strides[out_idx] = 0; // zero-stride stretch trick
            rhs_b_strides[out_idx] = stride_rhs;
        } else if (dim_rhs == 1) {
            out_shape[out_idx] = dim_lhs;
            lhs_b_strides[out_idx] = stride_lhs;
            rhs_b_strides[out_idx] = 0; // zero-stride stretch trick
        } else {
            throw std::invalid_argument("tensor shapes are non-broadcastable");
        }
    }
}

size_t Tensor::get_flat_index_from_broadcast(
    const std::vector<size_t>& current_coords,
    const std::vector<size_t>& broadcast_strides
){
    // computes the dot product between current multi-dimensional loop
    // coordinates and specialized broadcast stride vector
    size_t flat_idx = 0;
    for(size_t i {0}; i < current_coords.size(); ++i){
        flat_idx += current_coords[i] * broadcast_strides[i];
    }
    return flat_idx;
}

bool Tensor::advance_coordinates(
    std::vector<size_t>& coords,
    const std::vector<size_t>& target_shape
){
    if(coords.empty()) return false;

    // scan right-to-left from the innermost dimension
    for(int d {static_cast<int>(target_shape.size()) - 1}; d >= 0; --d){
        coords[d]++;
        if(coords[d] < target_shape[d]){
            return true; // increment suceeded without rolling over the edge
        }
        coords[d] = 0; // rollover this axis and carry the addition to the left
    }
    return false; // the entire tensor space has been fully traversed
}

// helper function to calculate all the boilerplate metadata
Tensor::ReductionMeta Tensor::prepare_reduction_metadata(size_t dim, bool keepdim) const {
    if (dim >= shape.size()) throw std::invalid_argument("Dimension out of bounds");

    ReductionMeta meta;
    meta.out_shape = this->shape;

    if (keepdim) {
        meta.out_shape[dim] = 1;
    } else {
        meta.out_shape.erase(meta.out_shape.begin() + dim);
    }

    if (meta.out_shape.empty()) {
        meta.out_shape = {1};
    }

    meta.total_out_elements = 1;
    for (size_t d : meta.out_shape) {
        meta.total_out_elements *= d;
    }

    meta.reduced_size = this->shape[dim];

    meta.inner_block_size = 1;
    for (size_t i = dim + 1; i < this->shape.size(); ++i) {
        meta.inner_block_size *= this->shape[i];
    }

    meta.outer_block_size = meta.total_out_elements / meta.inner_block_size;

    return meta;
}

// helper to create and reuse global cuBLAS handle
cublasHandle_t get_cublas_handle() {
    thread_local static cublasHandle_t handle = nullptr;
    if (handle == nullptr) CUBLAS_CHECK(cublasCreate(&handle));
    return handle;
}

// ------------------------------------------


// some kernels

// -------------------------------------
// UNARY KERNELS 
// -------------------------------------
// -------- forward pass kernels for GPU
// -------------------------------------

// --- relu

// functor for ReLU on the GPU
struct reluForwardOp {
    __device__ double operator()(double x) const {
        return x > 0.0  ? x : 0.0;
    }
};

// functor for ReLU's derivative
struct reluBackwardOp {
    __device__ double operator()(double x) const {
        return x > 0.0 ? 1.0 : 0.0;
    }
};


// --- exp

// functor for exp on the GPU
struct expForwardOp{
    __device__ double operator()(double x) const {
        return exp(x);
    }
};

// functor for exp's derivative
struct expBackwardOp {
    __device__ double operator()(double x) const {
        return exp(x);
    }
};


// --- tanh

// functor for tanh on the gpu
struct tanhForwardOp{
    __device__ double operator()(double x) const {
        return tanh(x);
    }
};

// functor for tanh's derivative
struct tanhBackwardOp {
    __device__ double operator()(double x) const {
        double t = tanh(x);
        return 1.0 - t * t;
    }
};


// --- sigmoid

// functor for sigmoid on the GPU
struct sigmoidForwardOp {
    __device__ double operator()(double x) const {
        return 1.0 / (1.0 + exp(-x));
    }
};

// functor for sigmoid's derivative
struct sigmoidBackwardOp {
    __device__ double operator()(double x) const {
        double s = 1.0 / (1.0 + exp(-x));
        return s * (1.0 - s);
    }
};


// --- log

// functor for log on GPU
struct logForwardOp {
    __device__ double operator()(double x) const {
        return log(x);
    }
};

// functor for log's derivative
struct logBackwardOp {
    __device__ double operator()(double x) const {
        return 1.0 / x;
    }
};


// --- pow

// functor for pow on GPU
struct powForwardOp{
    double exponent;
    powForwardOp(double exp) : exponent(exp){}

    __device__ double operator()(double x) const {
        return pow(x, exponent);
    }
};

// functor for pow's derivative
struct powBackwardOp {
    double exponent;
    powBackwardOp(double exp) : exponent(exp) {}

    __device__ double operator()(double x) const {
        return exponent * pow(x, exponent - 1.0);
    }
};


// --- sqrt

// functor for sqrt
struct sqrtForwardOp{
    __device__ double operator()(double x) const {
        return sqrt(x);
    }
};

// functor for sqrt's derivative
struct sqrtBackwardOp {
    __device__ double operator()(double x) const {
        return (x == 0.0) ? 0.0 : (0.5 / sqrt(x));
    }
};


// --- neg

// functor for neg
struct negForwardOp {
    __device__ double operator()(double x) const {
        return -x;
    }
};

// functor for neg's derivative
struct negBackwardOp {
    __device__ double operator()(double x) const {
        return -1.0;
    }
};


// --- templates

// unified __global__ kernel
template <typename Op>
__global__ void unary_forward(const double* input, double* output, size_t total_elements, Op op) {
    // calculate idx
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    // ensure we dont read/write past the end of array
    if(idx < total_elements){
        output[idx] = op(input[idx]);
    }
}

// unified __global__ backward kernel
template <typename Op>
__global__ void unary_backward(
    const double* upstream_grad, 
    double* self_grad, 
    const double* input_data, 
    size_t total_elements, 
    Op op
){
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < total_elements){
        // chain rule: local derivative * upstream grad
        self_grad[idx] += upstream_grad[idx] * op(input_data[idx]);
    }
}


// end of unary kernels....
// -------------------------------------------


// BINARY OPERATORS KERNELS
// ---------------------------------
constexpr size_t MAX_DIMS = 8;

struct BroadcastMeta {
    size_t ndim;
    size_t out_shape[MAX_DIMS];
    size_t lhs_strides[MAX_DIMS];
    size_t rhs_strides[MAX_DIMS];
};

// helper function to build BroadcastMeta on CPU
inline BroadcastMeta create_broadcast_meta(
    const std::vector<size_t>& out_shape,
    const std::vector<size_t>& lhs_b_strides,
    const std::vector<size_t>& rhs_b_strides
) {
    if (out_shape.size() > MAX_DIMS) {
        throw std::invalid_argument("Broadcasting exceeds maximum supported rank of 8 dimensions.");
    }
    BroadcastMeta meta;
    meta.ndim = out_shape.size();
    for (size_t i = 0; i < meta.ndim; ++i) {
        meta.out_shape[i] = out_shape[i];
        meta.lhs_strides[i] = lhs_b_strides[i];
        meta.rhs_strides[i] = rhs_b_strides[i];
    }
    return meta;
}


// ----- kernels

// --- additions (+)
struct addForwardOp {
    __device__ double operator()(double x, double y) const {
        return x + y;
    }
};

// functors for derivative of operator+
struct addLhsGradOp {
    __device__ double operator()(double x, double y) const {
        return 1.0;
    }
};

struct addRhsGradOp {
    __device__ double operator()(double x, double y) const {
        return 1.0;
    }
};


// --- substraction (-)
struct subForwardOp {
    __device__ double operator()(double x, double y) const {
        return x - y;
    }
};

// functors for derivative of operator-
struct subLhsGradOp{
    __device__ double operator()(double x, double y) const {
        return 1.0;
    }
};

struct subRhsGradOp {
    __device__ double operator()(double x, double y) const {
        return -1.0;
    }
};

// --- multiplier (*)



// ---- templates
// forward pass template
template <typename Op>
__global__ void binary_forward(const double* lhs, const double* rhs, double* output, size_t total_elements, Op op) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < total_elements) {
        output[idx] = op(lhs[idx], rhs[idx]);
    }
}

// forward pass with broadcasting
template <typename Op>
__global__ void binary_forward_broadcast(
    const double* lhs,
    const double* rhs,
    double* output,
    size_t total_elements,
    BroadcastMeta meta,
    Op op
) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < total_elements) {
        size_t flat_lhs = 0;
        size_t flat_rhs = 0;
        size_t temp = idx;


        for (int d = static_cast<int>(meta.ndim) - 1; d >= 0; --d) {
            size_t coord = temp % meta.out_shape[d];
            flat_lhs += coord * meta.lhs_strides[d];
            flat_rhs += coord * meta.rhs_strides[d];
            temp /= meta.out_shape[d];
        }

        output[idx] = op(lhs[flat_lhs], rhs[flat_rhs]);
    }
}

// ----

// backward pass template
// (matching shape)
template <typename Op>
__global__ void binary_backward(
    double* target_grad,
    const double* upstream_grad,
    const double* lhs,
    const double* rhs,
    size_t total_elements,
    Op local_grad_op
) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < total_elements){
        target_grad[idx] += upstream_grad[idx] * local_grad_op(lhs[idx], rhs[idx]);
    }
}

// with broadcasting
template <typename Op>
__global__ void binary_backward_broadcast(
    double* target_grad,
    const double* upstream_grad,
    const double* lhs,
    const double* rhs,
    size_t total_elements,
    BroadcastMeta meta,
    bool is_lhs,
    Op local_grad_op
) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < total_elements) {
        size_t flat_lhs = 0;
        size_t flat_rhs = 0;
        size_t temp = idx;

        for (int d = static_cast<int>(meta.ndim) - 1; d >= 0; --d) {
            size_t coord = temp % meta.out_shape[d];
            flat_lhs += coord * meta.lhs_strides[d];
            flat_rhs += coord * meta.rhs_strides[d];
            temp /= meta.out_shape[d];
        }

        size_t target_flat = is_lhs ? flat_lhs : flat_rhs;
        double local_grad  = local_grad_op(lhs[flat_lhs], rhs[flat_rhs]);

        atomicAdd(&target_grad[target_flat], upstream_grad[idx] * local_grad);
    }
}

// end of binary op kernels....
// ---------------------------------


// --------------
// some CUDA configs

constexpr int threads = 256;

// -----------------------------------
// unary kernel launchers
// unary forward pass kernel launcher
template <typename Op>
inline void launch_unary_forward(
    const double* input, 
    double* output, 
    size_t total_elements, 
    Op op
) {
    int blocks = cuda_utils::ceil_div(total_elements, threads);
    unary_forward<<<blocks, threads>>>(input, output, total_elements, op);
    CUDA_CHECK(cudaGetLastError());
}

// unary backward pass kernel launcher
template <typename Op>
inline void launch_unary_backward(
    const double* upstream_grad, 
    double* self_grad, 
    const double* input_data, 
    size_t total_elements, 
    Op op
) {
    int blocks = cuda_utils::ceil_div(total_elements, threads);
    unary_backward<<<blocks, threads>>>(upstream_grad, self_grad, input_data, total_elements, op);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------
// binary kernel launchers
// binary forward kernel launcher
template <typename Op>
inline void launch_binary_forward(
    const double* lhs,
    const double* rhs,
    double* output,
    size_t total_elements,
    Op op
) {
    int blocks = cuda_utils::ceil_div(total_elements, threads);
    binary_forward<<<blocks, threads>>>(lhs, rhs, output, total_elements, op);
    CUDA_CHECK(cudaGetLastError());
}

// binary forward with broadcasting launcher
template <typename Op>
inline void launch_binary_forward_broadcast(
    const double* lhs,
    const double* rhs,
    double* output,
    size_t total_elements,
    BroadcastMeta meta,
    Op op
) {
    int blocks = cuda_utils::ceil_div(total_elements, threads);
    binary_forward_broadcast<<<blocks, threads>>>(lhs, rhs, output, total_elements, meta, op);
    CUDA_CHECK(cudaGetLastError());
}

// binary backward (matching shape)
template <typename Op>
inline void launch_binary_backward(
    double* target_grad,
    const double* upstream_grad,
    const double* lhs,
    const double* rhs,
    size_t total_elements,
    Op local_grad_op
) {
    int blocks = cuda_utils::ceil_div(total_elements, threads);
    binary_backward<<<blocks, threads>>>(target_grad, upstream_grad, lhs, rhs, total_elements, local_grad_op);
    CUDA_CHECK(cudaGetLastError());
}

// binary backward with broadcasting
template <typename Op>
inline void launch_binary_backward_broadcast(
    double* target_grad,
    const double* upstream_grad,
    const double* lhs,
    const double* rhs,
    size_t total_elements,
    BroadcastMeta meta,
    bool is_lhs,
    Op local_grad_op
) {
    int blocks = cuda_utils::ceil_div(total_elements, threads);
    binary_backward_broadcast<<<blocks, threads>>>(target_grad, upstream_grad, lhs, rhs, total_elements, meta, is_lhs, local_grad_op);
    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------

void Tensor::ensure_grad_allocated(){
    if(grad == nullptr){
        grad = (std::make_shared<std::vector<double>>(data->size(), 0.0));

        // alloc in GPU if on GPU
        if (device == Device::CUDA){
            size_t bytes = data->size() * sizeof(double);
            CUDA_CHECK(cudaMalloc(&cuda_grad, bytes));
            // copy the zeros from cpu
            CUDA_CHECK(cudaMemcpy(cuda_grad, grad->data(), bytes, cudaMemcpyHostToDevice));
        }
    }
}

// verifies if row-major order perfectly aligns with standard memory sequences
bool Tensor::is_contiguous() const {
    size_t expected_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (shape[i] != 1 && strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= shape[i];
    }
    return true;
}

// deep-copies data layouts to establish standard physical contiguity
TensorPtr Tensor::contiguous() {
    if (is_contiguous()) {
        return shared_from_this();
    }
    
    // calculate logical elements from shape instead of physical data footprint
    size_t total_elements = 1;
    for (size_t s : shape) total_elements *= s;
    
    std::vector<double> contiguous_values(total_elements);
    // stateless coordinate resolution for safe parallelization
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(total_elements); ++i) {
        size_t flat_idx = 0;
        size_t temp = i;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
            size_t coord = temp % shape[d];
            flat_idx += coord * strides[d];
            temp /= shape[d];
        }
        contiguous_values[i] = (*data)[flat_idx];
    }
    
    auto out = std::make_shared<Tensor>(contiguous_values, shape, std::vector<TensorPtr>{shared_from_this()}, "contiguous");
    
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();
    out->backward_func = [self, weak_out, total_elements]() {
        if (auto out_ptr = weak_out.lock()) {
            std::vector<size_t> back_coords(self->shape.size(), 0);
            for (size_t i = 0; i < total_elements; ++i) {
                size_t flat_idx = self->get_flat_index(back_coords);
                (*self->grad)[flat_idx] += (*out_ptr->grad)[i];
                advance_coordinates(back_coords, self->shape);
            }
        }
    };
    return out;
}

// zeroes the internal gradient vector array
void Tensor::zero_grad() {
    if (grad) {
        std::fill(grad->begin(), grad->end(), 0.0);
    }
}



// ---------------------------
// BINARY OPERATORS
// ---------------------------

// in-place addition supporting standard fast-paths and multi-dimensional coordinate loops
TensorPtr Tensor::add_(const TensorPtr& other) {
    // copy-on-write layout materialization to protect shared views and autograd history
    if (!is_contiguous() || data.use_count() > 1) {
        size_t total_elements = 1;
        for (size_t s : shape) total_elements *= s;
        
        std::vector<double> contiguous_values(total_elements);
        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            size_t flat_idx = 0;
            size_t temp = i;
            for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                size_t coord = temp % shape[d];
                flat_idx += coord * strides[d];
                temp /= shape[d];
            }
            contiguous_values[i] = (*data)[flat_idx];
        }
        
        data = std::make_shared<std::vector<double>>(contiguous_values);
        
        strides.resize(shape.size(), 1);
        if (!shape.empty()) {
            for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
    }

    if (is_contiguous() && other->is_contiguous() && shape == other->shape) {
        for (size_t i = 0; i < data->size(); ++i) {
            (*data)[i] += (*other->data)[i];
        }
    } else {
        std::vector<size_t> out_shape;
        std::vector<size_t> lhs_b_strides;
        std::vector<size_t> rhs_b_strides;
        compute_broadcast_metadata(shared_from_this(), other, out_shape, lhs_b_strides, rhs_b_strides);
        
        if (out_shape != this->shape) {
            throw std::invalid_argument("In-place targets cannot be expanded via broadcasting.");
        }
        
        size_t total_elements = data->size();
        
        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            size_t flat_lhs = 0;
            size_t flat_rhs = 0;
            size_t temp = i;
            for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
                size_t coord = temp % out_shape[d];
                flat_lhs += coord * lhs_b_strides[d];
                flat_rhs += coord * rhs_b_strides[d];
                temp /= out_shape[d];
            }
            (*data)[flat_lhs] += (*other->data)[flat_rhs];
        }
    }
    return shared_from_this();
}

// in-place substraction
TensorPtr Tensor::sub_(const TensorPtr& other) {
    // copy-on-write layout materialization to protect shared views and autograd history
    if (!is_contiguous() || data.use_count() > 1) {
        size_t total_elements = 1;
        for (size_t s : shape) total_elements *= s;
        
        std::vector<double> contiguous_values(total_elements);
        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            size_t flat_idx = 0;
            size_t temp = i;
            for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                size_t coord = temp % shape[d];
                flat_idx += coord * strides[d];
                temp /= shape[d];
            }
            contiguous_values[i] = (*data)[flat_idx];
        }
        
        data = std::make_shared<std::vector<double>>(contiguous_values);
        
        strides.resize(shape.size(), 1);
        if (!shape.empty()) {
            for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
    }

    if (is_contiguous() && other->is_contiguous() && shape == other->shape) {
        for (size_t i = 0; i < data->size(); ++i) {
            (*data)[i] -= (*other->data)[i];
        }
    } else {
        std::vector<size_t> out_shape;
        std::vector<size_t> lhs_b_strides;
        std::vector<size_t> rhs_b_strides;
        compute_broadcast_metadata(shared_from_this(), other, out_shape, lhs_b_strides, rhs_b_strides);
        
        if (out_shape != this->shape) {
            throw std::invalid_argument("In-place targets cannot be expanded via broadcasting.");
        }
        
        size_t total_elements = data->size();
        
        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            size_t flat_lhs = 0;
            size_t flat_rhs = 0;
            size_t temp = i;
            for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
                size_t coord = temp % out_shape[d];
                flat_lhs += coord * lhs_b_strides[d];
                flat_rhs += coord * rhs_b_strides[d];
                temp /= out_shape[d];
            }
            (*data)[flat_lhs] -= (*other->data)[flat_rhs];
        }
    }
    return shared_from_this();
}

// addition op
TensorPtr operator+(const TensorPtr& lhs, const TensorPtr& rhs){
    if (lhs->device != rhs->device) throw std::invalid_argument("Tensors must be on the same device");

    std::vector<size_t> out_shape;
    std::vector<size_t> lhs_b_strides;
    std::vector<size_t> rhs_b_strides;

    // calculate broadcast shapes and zero-strides properties
    Tensor::compute_broadcast_metadata(lhs, rhs, out_shape, lhs_b_strides, rhs_b_strides);

    // calculate total elements needed for the output data array
    size_t total_elements = 1;
    for(size_t dim : out_shape) total_elements *= dim;
    
    // determine device target
    Device active_device = (lhs->device == Device::CUDA || rhs->device == Device::CUDA) ? Device::CUDA : Device::CPU;

    std::vector<double> dummy_vals(total_elements);
    // construct graph node
    auto out = std::make_shared<Tensor>(std::move(dummy_vals), out_shape, std::vector<TensorPtr>{lhs, rhs}, "+");
    out->to(active_device);

    bool is_matching_shape = (lhs->shape == rhs->shape) && lhs->is_contiguous() && rhs->is_contiguous();

    // forward pass
    if(active_device == Device::CUDA) {
        // --- gpu forward
        if(is_matching_shape){
            launch_binary_forward(
                lhs->cuda_data,
                rhs->cuda_data,
                out->cuda_data,
                total_elements,
                addForwardOp()
            );

        } else {
            // --- gpu forward with broadcasting
            BroadcastMeta meta = create_broadcast_meta(out_shape, lhs_b_strides, rhs_b_strides);
            launch_binary_forward_broadcast(
                lhs->cuda_data,
                rhs->cuda_data,
                out->cuda_data,
                total_elements,
                meta,
                addForwardOp()
            );

        }
    } else {
        // --- cpu forward
        double* out_ptr = out->data->data();
        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            size_t flat_lhs = 0;
            size_t flat_rhs = 0;
            size_t temp = i;
            for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
                size_t coord = temp % out_shape[d];
                flat_lhs += coord * lhs_b_strides[d];
                flat_rhs += coord * rhs_b_strides[d];
                temp /= out_shape[d];
            }
            out_ptr[i] = (*lhs->data)[flat_lhs] + (*rhs->data)[flat_rhs];
        }
    }


    // --- backward pass
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, out_shape, lhs_b_strides, rhs_b_strides, total_elements, active_device, is_matching_shape]() {
        if(auto out_ptr = weak_out.lock()) {

            if(active_device == Device::CUDA){
                // --- GPU backward pass
                if(is_matching_shape){
                    if (lhs->requires_grad) {
                        launch_binary_backward(
                            lhs->cuda_grad, 
                            out_ptr->cuda_grad, 
                            lhs->cuda_data, 
                            rhs->cuda_data, 
                            total_elements, 
                            addLhsGradOp()
                        );
                    }
                    if (rhs->requires_grad) {
                        launch_binary_backward(
                            rhs->cuda_grad, 
                            out_ptr->cuda_grad, 
                            lhs->cuda_data, 
                            rhs->cuda_data, 
                            total_elements, 
                            addRhsGradOp()
                        );
                    }

                } else {
                    // --- backward pass with broadcasting
                    BroadcastMeta meta = create_broadcast_meta(out_shape, lhs_b_strides, rhs_b_strides);
                    if(lhs->requires_grad){
                        launch_binary_backward_broadcast(
                            lhs->cuda_grad, 
                            out_ptr->cuda_grad, 
                            lhs->cuda_data, 
                            rhs->cuda_data, 
                            total_elements, 
                            meta, 
                            true, 
                            addLhsGradOp()
                        );
                    }

                    if(rhs->requires_grad){
                        launch_binary_backward_broadcast(
                            rhs->cuda_grad, 
                            out_ptr->cuda_grad, 
                            lhs->cuda_data, 
                            rhs->cuda_data, 
                            total_elements, 
                            meta, 
                            false, 
                            addRhsGradOp()
                        );
                    }
                }

            } else {
                std::vector<size_t> back_idx(out_shape.size(), 0);
    
                for(size_t i {0}; i < total_elements; ++i){
                    size_t flat_lhs = Tensor::get_flat_index_from_broadcast(back_idx, lhs_b_strides);
                    size_t flat_rhs = Tensor::get_flat_index_from_broadcast(back_idx, rhs_b_strides);
                    
                    // upstream gradient vector aligns perfectly with total_elements sequence
                    double upstream_grad = (*out_ptr->grad)[i];
    
                    // zero-stride values automatically combine multi-dimensional gradients here
                    if(lhs->requires_grad) (*lhs->grad)[flat_lhs] += upstream_grad;
                    if(rhs->requires_grad) (*rhs->grad)[flat_rhs] += upstream_grad;
    
                    // step backward coordinate tracking layout forward
                    Tensor::advance_coordinates(back_idx, out_shape);
                }
            }
            
        }
    };

    return out;
}

// substraction op
TensorPtr operator-(const TensorPtr& lhs, const TensorPtr& rhs){
    if (lhs->device != rhs->device) throw std::invalid_argument("Tensors must be on the same device");

    std::vector<size_t> out_shape;
    std::vector<size_t> lhs_b_strides;
    std::vector<size_t> rhs_b_strides;

    // calculate broadcast shapes and zero-strides properties
    Tensor::compute_broadcast_metadata(lhs, rhs, out_shape, lhs_b_strides, rhs_b_strides);

    // calc total elements needed for out data
    size_t total_elements = 1;
    for(size_t dim : out_shape) total_elements *= dim;

    // determine device target
    Device active_device = (lhs->device == Device::CUDA || rhs->device == Device::CUDA) ? Device::CUDA : Device::CPU;

    std::vector<double> dummy_vals(total_elements);
    // construct graph node
    auto out = std::make_shared<Tensor>(std::move(dummy_vals), out_shape, std::vector<TensorPtr>{lhs,rhs}, "-");
    out->to(active_device);

    bool is_matching_shape = (lhs->shape == rhs->shape) && lhs->is_contiguous() && rhs->is_contiguous();

    // forward pass
    if(active_device == Device::CUDA){
        // --- gpu forward
        if(is_matching_shape){
            launch_binary_forward(
                lhs->cuda_data,
                rhs->cuda_data,
                out->cuda_data,
                total_elements,
                subForwardOp()
            );

        } else {
            // --- gpu forward with broadcasting
            BroadcastMeta meta = create_broadcast_meta(out_shape, lhs_b_strides, rhs_b_strides);
            launch_binary_forward_broadcast(
                lhs->cuda_data,
                rhs->cuda_data,
                out->cuda_data,
                total_elements,
                meta,
                subForwardOp()
            );

        }
    } else {
        // --- cpu forward pass
        double* out_ptr = out->data->data();
        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            size_t flat_lhs = 0;
            size_t flat_rhs = 0;
            size_t temp = i;
            for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
                size_t coord = temp % out_shape[d];
                flat_lhs += coord * lhs_b_strides[d];
                flat_rhs += coord * rhs_b_strides[d];
                temp /= out_shape[d];
            }
            out_ptr[i] = (*lhs->data)[flat_lhs] - (*rhs->data)[flat_rhs];
        }
    }


    // --- backward pass
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, out_shape, lhs_b_strides, rhs_b_strides, total_elements, active_device, is_matching_shape]() {
        if(auto out_ptr = weak_out.lock()){

            if(active_device == Device::CUDA){
                // --- GPU backward pass
                if(is_matching_shape) {
                    if(lhs->requires_grad){
                        launch_binary_backward(
                            lhs->cuda_grad, 
                            out_ptr->cuda_grad, 
                            lhs->cuda_data, 
                            rhs->cuda_data, 
                            total_elements, 
                            subLhsGradOp()
                        );
                    }
                    if(rhs->requires_grad){
                        launch_binary_backward(
                            rhs->cuda_grad, 
                            out_ptr->cuda_grad, 
                            lhs->cuda_data, 
                            rhs->cuda_data, 
                            total_elements, 
                            subRhsGradOp()
                        );
                    }

                } else {
                    // --- backward pass with broadcasting
                    BroadcastMeta meta = create_broadcast_meta(out_shape, lhs_b_strides, rhs_b_strides);
                    if(lhs->requires_grad){
                        launch_binary_backward_broadcast(
                            lhs->cuda_grad, 
                            out_ptr->cuda_grad, 
                            lhs->cuda_data, 
                            rhs->cuda_data, 
                            total_elements, 
                            meta, 
                            true, 
                            subLhsGradOp()
                        );
                    }

                    if(rhs->requires_grad){
                        launch_binary_backward_broadcast(
                            rhs->cuda_grad, 
                            out_ptr->cuda_grad, 
                            lhs->cuda_data, 
                            rhs->cuda_data, 
                            total_elements, 
                            meta, 
                            false, 
                            subRhsGradOp()
                        );
                    }
                }

            }  else {
                std::vector<size_t> back_idx(out_shape.size(), 0);
    
                for(size_t i {0}; i < total_elements; ++i){
                    size_t flat_lhs = Tensor::get_flat_index_from_broadcast(back_idx, lhs_b_strides);
                    size_t flat_rhs = Tensor::get_flat_index_from_broadcast(back_idx, rhs_b_strides);
                    
                    // upstream gradient vector aligns perfectly with total_elements sequence
                    auto upstream_grad = (*out_ptr->grad)[i];
    
                    // zero-stride values automatically combine multi-dimensional gradients here
                    if(lhs->requires_grad) (*lhs->grad)[flat_lhs] += upstream_grad;
                    if(rhs->requires_grad) (*rhs->grad)[flat_rhs] -= upstream_grad;
    
                    // step backward coordinate tracking layout forward
                    Tensor::advance_coordinates(back_idx, out_shape);
                }
            }
            
        }
    };

    return out;
}

// element wise operator
TensorPtr operator*(const TensorPtr& lhs, const TensorPtr& rhs){
    std::vector<size_t> out_shape;
    std::vector<size_t> lhs_b_strides;
    std::vector<size_t> rhs_b_strides;

    // calculate broadcast shapes and zero-strides properties
    Tensor::compute_broadcast_metadata(lhs, rhs, out_shape, lhs_b_strides, rhs_b_strides);

    // calculate total elements needed for the output data array
    size_t total_elements = 1;
    for(size_t dim : out_shape){
        total_elements *= dim;
    }
    std::vector<double> out_values(total_elements, 0.0);

    // forward pass
    // stateless coordinate resolution for safe parallelization
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(total_elements); ++i) {
        size_t flat_lhs = 0;
        size_t flat_rhs = 0;
        size_t temp = i;
        for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
            size_t coord = temp % out_shape[d];
            flat_lhs += coord * lhs_b_strides[d];
            flat_rhs += coord * rhs_b_strides[d];
            temp /= out_shape[d];
        }
        out_values[i] = (*lhs->data)[flat_lhs] * (*rhs->data)[flat_rhs];
    }

    auto out = std::make_shared<Tensor>(out_values, out_shape, std::vector<TensorPtr>{lhs, rhs}, "*");

    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, out_shape, lhs_b_strides, rhs_b_strides, total_elements](){
        if (auto out_ptr = weak_out.lock()){
            std::vector<size_t> back_idx(out_shape.size(), 0);

            for(size_t i {0}; i < total_elements; ++i){
                size_t flat_lhs = Tensor::get_flat_index_from_broadcast(back_idx, lhs_b_strides);
                size_t flat_rhs = Tensor::get_flat_index_from_broadcast(back_idx, rhs_b_strides);
                
                auto upstream_grad = (*out_ptr->grad)[i];

                if(lhs->requires_grad) (*lhs->grad)[flat_lhs] += upstream_grad * (*rhs->data)[flat_rhs];
                if(rhs->requires_grad) (*rhs->grad)[flat_rhs] += upstream_grad * (*lhs->data)[flat_lhs];

                Tensor::advance_coordinates(back_idx, out_shape);
            }
        }
    };

    return out;
}

// tensor * scalar ops
TensorPtr operator*(const TensorPtr& lhs, double rhs) {
    size_t total_elements = 1;
    for (size_t s : lhs->shape) total_elements *= s;

    std::vector<double> out_values(total_elements);
    // stateless coordinate resolution for safe parallelization
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(total_elements); ++i) {
        size_t flat_idx = 0;
        size_t temp = i;
        for (int d = static_cast<int>(lhs->shape.size()) - 1; d >= 0; --d) {
            size_t coord = temp % lhs->shape[d];
            flat_idx += coord * lhs->strides[d];
            temp /= lhs->shape[d];
        }
        out_values[i] = (*lhs->data)[flat_idx] * rhs;
    }

    auto out = std::make_shared<Tensor>(out_values, lhs->shape, std::vector<TensorPtr>{lhs}, "*");

    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, total_elements]() {
        if (auto out_ptr = weak_out.lock()) {
            if (lhs->requires_grad) {
                std::vector<size_t> back_coords(lhs->shape.size(), 0);
                for (size_t i = 0; i < total_elements; ++i) {
                    size_t flat_idx = lhs->get_flat_index(back_coords);
                    (*lhs->grad)[flat_idx] += (*out_ptr->grad)[i] * rhs;
                    Tensor::advance_coordinates(back_coords, lhs->shape);
                }
            }
        }
    };

    return out;
}

TensorPtr operator*(double lhs, const TensorPtr& rhs) {
    return rhs * lhs;
}

// division op
TensorPtr operator/(const TensorPtr& lhs, const TensorPtr& rhs){
    std::vector<size_t> out_shape;
    std::vector<size_t> lhs_b_strides;
    std::vector<size_t> rhs_b_strides;

    // calculate broadcast shapes and zero-strides properties
    Tensor::compute_broadcast_metadata(lhs, rhs, out_shape, lhs_b_strides, rhs_b_strides);

    // calculate total elements needed for the output data array
    size_t total_elements = 1;
    for (size_t dim : out_shape) {
        total_elements *= dim;
    }
    std::vector<double> out_values(total_elements, 0.0);

    // forward pass
    // stateless coordinate resolution for safe parallelization
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(total_elements); ++i) {
        size_t flat_lhs = 0;
        size_t flat_rhs = 0;
        size_t temp = i;
        for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
            size_t coord = temp % out_shape[d];
            flat_lhs += coord * lhs_b_strides[d];
            flat_rhs += coord * rhs_b_strides[d];
            temp /= out_shape[d];
        }
        // standard element-wise division
        out_values[i] = (*lhs->data)[flat_lhs] / (*rhs->data)[flat_rhs];
    }

    auto out = std::make_shared<Tensor>(out_values, out_shape, std::vector<TensorPtr>{lhs, rhs}, "/");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, out_shape, lhs_b_strides, rhs_b_strides, total_elements]() {
        if (auto out_ptr = weak_out.lock()) {
            std::vector<size_t> back_idx(out_shape.size(), 0);

            for (size_t i = 0; i < total_elements; ++i) {
                size_t flat_lhs = Tensor::get_flat_index_from_broadcast(back_idx, lhs_b_strides);
                size_t flat_rhs = Tensor::get_flat_index_from_broadcast(back_idx, rhs_b_strides);
                
                auto upstream_grad = (*out_ptr->grad)[i];
                double lhs_val = (*lhs->data)[flat_lhs];
                double rhs_val = (*rhs->data)[flat_rhs];

                // d(x/y) / dx = 1 / y
                if(lhs->requires_grad) (*lhs->grad)[flat_lhs] += upstream_grad / rhs_val;
                
                // d(x/y) / dy = -x / y^2
                if(rhs->requires_grad) (*rhs->grad)[flat_rhs] -= upstream_grad * (lhs_val / (rhs_val * rhs_val));

                Tensor::advance_coordinates(back_idx, out_shape);
            }
        }
    };

    return out;
}

// matmul op
TensorPtr Tensor::matmul(const TensorPtr& lhs, const TensorPtr& rhs){
    // verify dimensions
    if (lhs->shape.size() < 2 || rhs->shape.size() < 2) {
        throw std::invalid_argument("both inputs to matmul must be at least 2D tensors.");
    }

    size_t lhs_rank = lhs->shape.size();
    size_t rhs_rank = rhs->shape.size();

    // isolate core matrix dimensions from the trailing two axes
    size_t M = lhs->shape[lhs_rank - 2];
    size_t K = lhs->shape[lhs_rank - 1];
    size_t rhs_K = rhs->shape[rhs_rank - 2];
    size_t N = rhs->shape[rhs_rank - 1];

    if (K != rhs_K) {
        throw std::invalid_argument("matrix inner dimensions must match for multiplication.");
    }

    // isolate leading batch dimensions
    size_t lhs_batch_dims = lhs_rank - 2;
    size_t rhs_batch_dims = rhs_rank - 2;
    size_t out_batch_dims = std::max(lhs_batch_dims, rhs_batch_dims);

    std::vector<size_t> batch_shape(out_batch_dims);
    std::vector<size_t> lhs_batch_strides(out_batch_dims);
    std::vector<size_t> rhs_batch_strides(out_batch_dims);

    // compute broadcasting metadata specifically for the batch dimensions
    for (size_t i = 0; i < out_batch_dims; ++i){
        size_t out_idx = out_batch_dims - 1 - i;
        size_t dim_lhs = (i < lhs_batch_dims) ? lhs->shape[lhs_batch_dims - 1 - i] : 1;
        size_t stride_lhs = (i < lhs_batch_dims) ? lhs->strides[lhs_batch_dims - 1 - i] : 0;
        size_t dim_rhs = (i < rhs_batch_dims) ? rhs->shape[rhs_batch_dims - 1 - i] : 1;
        size_t stride_rhs = (i < rhs_batch_dims) ? rhs->strides[rhs_batch_dims - 1 - i] : 0;

        if (dim_lhs == dim_rhs) {
            batch_shape[out_idx] = dim_lhs;
            lhs_batch_strides[out_idx] = stride_lhs;
            rhs_batch_strides[out_idx] = stride_rhs;
        } else if (dim_lhs == 1) {
            batch_shape[out_idx] = dim_rhs;
            lhs_batch_strides[out_idx] = 0;
            rhs_batch_strides[out_idx] = stride_rhs;
        } else if (dim_rhs == 1) {
            batch_shape[out_idx] = dim_lhs;
            lhs_batch_strides[out_idx] = stride_lhs;
            rhs_batch_strides[out_idx] = 0;
        } else {
            throw std::invalid_argument("tensor batch shapes are non-broadcastable.");
        }
    }

    // assemble unified output shape configuration
    std::vector<size_t> out_shape = batch_shape;
    out_shape.push_back(M);
    out_shape.push_back(N);

    size_t total_batches = 1;
    for (size_t dim : batch_shape) total_batches *= dim;

    // create output tensor and push to target device
    std::vector<double> dummy_vals(total_batches * M * N);
    auto out = std::make_shared<Tensor>(std::move(dummy_vals), std::move(out_shape), std::vector<TensorPtr>{lhs, rhs}, "matmul");
    out->to(lhs->device);

    // extract invariant trailing strides for 2D sub-matrix steps
    size_t lhs_stride_M = lhs->strides[lhs_rank - 2];
    size_t lhs_stride_K = lhs->strides[lhs_rank - 1];
    size_t rhs_stride_K = rhs->strides[rhs_rank - 2];
    size_t rhs_stride_N = rhs->strides[rhs_rank - 1];

    if(lhs->device == Device::CUDA){
        // cuBLAS gpu forward pass
        cublasHandle_t handle = get_cublas_handle();
        const double alpha = 1.0;
        const double beta = 0.0;

        for(size_t b = 0; b < total_batches; ++b){
            size_t batch_lhs_off = 0, batch_rhs_off = 0;
            size_t temp = b;
            for(int d = static_cast<int>(out_batch_dims) - 1; d >= 0; --d){
                size_t coord = temp % batch_shape[d];
                batch_lhs_off += coord * lhs_batch_strides[d];
                batch_rhs_off += coord * rhs_batch_strides[d];
                temp /= batch_shape[d];
            }
            size_t batch_out_off = b * M * N;

            // swapped operands (rhs, lhs) for row-major/col-major tricks
            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        rhs->cuda_data + batch_rhs_off, N,
                        lhs->cuda_data + batch_lhs_off, K,
                        &beta,
                        out->cuda_data + batch_out_off, N));
        }
    } else { //CPU forward pass
        const double* lhs_ptr = lhs->data->data();
        const double* rhs_ptr = rhs->data->data();
        double* out_ptr = out->data->data(); 
    
        // forward pass now uses explicit index reconstruction to isolate thread state
        #pragma omp parallel for schedule(static)
        for (size_t b = 0; b < total_batches; ++b) {
            size_t batch_lhs_off = 0;
            size_t batch_rhs_off = 0;
            size_t temp = b;
    
            // reconstruct strides directly using stack primitives
            for (int d = static_cast<int>(out_batch_dims) - 1; d >= 0; --d) {
                size_t coord = temp % batch_shape[d];
                batch_lhs_off += coord * lhs_batch_strides[d];
                batch_rhs_off += coord * rhs_batch_strides[d];
                temp /= batch_shape[d];
            }
    
            size_t batch_out_off = b * M * N;
    
            // standard 2D matrix multiplication multiplication on the current batch slice
            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    double sum_val = 0.0;
                    for (size_t k = 0; k < K; ++k) {
                        sum_val += lhs_ptr[batch_lhs_off + i * lhs_stride_M + k * lhs_stride_K] * 
                                   rhs_ptr[batch_rhs_off + k * rhs_stride_K + j * rhs_stride_N];
                    }
                    out_ptr[batch_out_off + i * N + j] = sum_val;
                }
            }
        }
    }

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, batch_shape, 
                          lhs_batch_strides, rhs_batch_strides, 
                          total_batches, M, K, N, 
                          lhs_stride_M, lhs_stride_K, rhs_stride_K, rhs_stride_N]() {

        if (auto out_ptr = weak_out.lock()) {
            std::vector<size_t> back_batch_idx(batch_shape.size(), 0);

            if(lhs->device == Device::CUDA) {
                // -------- GPU backward pass using cuBLAS
                cublasHandle_t handle = get_cublas_handle();
                const double alpha = 1.0;
                const double beta = 1.0; // grad must accumulate

                for(size_t b = 0; b < total_batches; ++b){
                    size_t batch_lhs_off = Tensor::get_flat_index_from_broadcast(back_batch_idx, lhs_batch_strides);
                    size_t batch_rhs_off = Tensor::get_flat_index_from_broadcast(back_batch_idx, rhs_batch_strides);
                    size_t batch_out_off = b * M * N;

                    // dL/dLHS = dL/dOut * RHS^T
                    if (lhs->requires_grad) {
                        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                    K, M, N,
                                    &alpha,
                                    rhs->cuda_data + batch_rhs_off, N,
                                    out_ptr->cuda_grad + batch_out_off, N,
                                    &beta,
                                    lhs->cuda_grad + batch_lhs_off, K));
                    }

                    // dL/dRHS = LHS^T * dL/dOut
                    if (rhs->requires_grad) {
                        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                    N, K, M,
                                    &alpha,
                                    out_ptr->cuda_grad + batch_out_off, N,
                                    lhs->cuda_data + batch_lhs_off, K,
                                    &beta,
                                    rhs->cuda_grad + batch_rhs_off, N));
                    }
                    Tensor::advance_coordinates(back_batch_idx, batch_shape);
                }
            } else { // ----- backward pass on CPU
                for (size_t b = 0; b < total_batches; ++b) {
                    size_t batch_lhs_off = Tensor::get_flat_index_from_broadcast(back_batch_idx, lhs_batch_strides);
                    size_t batch_rhs_off = Tensor::get_flat_index_from_broadcast(back_batch_idx, rhs_batch_strides);
                    size_t batch_out_off = b * M * N;
    
                    // dL/dLHS = dL/dOut * RHS^T
                    if(lhs->requires_grad){
                        for (size_t i = 0; i < M; ++i) {
                            for (size_t k = 0; k < K; ++k) {
                                double grad_sum = 0.0;
                                for (size_t j = 0; j < N; ++j) {
                                    size_t out_flat = batch_out_off + i * N + j;
                                    size_t rhs_flat = batch_rhs_off + k * rhs_stride_K + j * rhs_stride_N;
                                    grad_sum += (*out_ptr->grad)[out_flat] * (*rhs->data)[rhs_flat];
                                }
                                size_t lhs_flat = batch_lhs_off + i * lhs_stride_M + k * lhs_stride_K;
                                (*lhs->grad)[lhs_flat] += grad_sum; // automatically reduces across broadcasted batch dimensions
                            }
                        }
                    }
    
                    // dL/dRHS = LHS^T * dL/dOut
                    if(rhs->requires_grad){
                        for (size_t k = 0; k < K; ++k) {
                            for (size_t j = 0; j < N; ++j) {
                                double grad_sum = 0.0;
                                for (size_t i = 0; i < M; ++i) {
                                    size_t out_flat = batch_out_off + i * N + j;
                                    size_t lhs_flat = batch_lhs_off + i * lhs_stride_M + k * lhs_stride_K;
                                    grad_sum += (*lhs->data)[lhs_flat] * (*out_ptr->grad)[out_flat];
                                }
                                size_t rhs_flat = batch_rhs_off + k * rhs_stride_K + j * rhs_stride_N;
                                (*rhs->grad)[rhs_flat] += grad_sum; // automatically reduces across broadcasted batch dimensions
                            }
                        }
                    }
    
                    Tensor::advance_coordinates(back_batch_idx, batch_shape);
                }
            }
        }
    };

    return out;
}

// end of binary Kernels...
// --------------------------------



//sum function
TensorPtr Tensor::sum() {
    // if non-contiguous, force contiguity to prevent physical memory misalignment
    auto active_this = is_contiguous() ? shared_from_this() : contiguous();

    std::vector<double> out_values(1, 0.0);
    for (double val : (*active_this->data)) {
        out_values[0] += val;
    }

    auto out = std::make_shared<Tensor>(out_values, std::vector<size_t>{1}, std::vector<TensorPtr>{active_this}, "sum");

    std::weak_ptr<Tensor> weak_out = out;

    out->backward_func = [active_this, weak_out]() {
        if (auto out_ptr = weak_out.lock()) {
            double upstream_grad = (*out_ptr->grad)[0];
            // Safe lazy gradient allocation loop guard
            if (active_this->requires_grad) {
                for (size_t i = 0; i < active_this->data->size(); ++i) {
                    (*active_this->grad)[i] += upstream_grad;
                }
            }
        }
    };

    return out;
}

// sum function with dimensions
TensorPtr Tensor::sum(size_t dim, bool keepdim){
    ReductionMeta meta = prepare_reduction_metadata(dim, keepdim);

    std::vector<double> out_vals(meta.total_out_elements, 0.0);

    // forward pass
    for(size_t outer {0}; outer < meta.outer_block_size; ++outer){
        for(size_t inner {0}; inner < meta.inner_block_size; ++inner){
            size_t out_flat = outer * meta.inner_block_size + inner;

            double sum = 0.0;

            for(size_t r {0}; r < meta.reduced_size; ++r){
                std::vector<size_t> coords(shape.size());
                coords[dim] = r;

                size_t temp_inner = inner;
                for (size_t d = shape.size() - 1; d > dim; --d) {
                    coords[d] = temp_inner % shape[d];
                    temp_inner /= shape[d];
                }
                size_t temp_outer = outer;
                for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
                    coords[d] = temp_outer % shape[d];
                    temp_outer /= shape[d];
                }

                size_t self_flat = get_flat_index(coords);
                sum += (*data)[self_flat];
            }

            out_vals[out_flat] = sum;
        }
    }

    auto out = std::make_shared<Tensor>(out_vals, meta.out_shape, std::vector<TensorPtr>{shared_from_this()}, "sum");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();

    size_t outer_bs = meta.outer_block_size;
    size_t inner_bs = meta.inner_block_size;
    size_t r_size = meta.reduced_size;

    out->backward_func = [self, weak_out, outer_bs, inner_bs, r_size, dim](){
        if(auto out_ptr = weak_out.lock()){

            for (size_t outer {0}; outer < outer_bs; ++outer) {
                for (size_t inner {0}; inner < inner_bs; ++inner) {
                    size_t out_flat = outer * inner_bs + inner;
                    double upstream_grad = (*out_ptr->grad)[out_flat];

                    if(self->requires_grad){
                        for (size_t r {0}; r < r_size; ++r) {
                            std::vector<size_t> back_coords(self->shape.size());
                            back_coords[dim] = r;
                            
                            size_t temp_inner = inner;
                            for (size_t d = self->shape.size() - 1; d > dim; --d) {
                                back_coords[d] = temp_inner % self->shape[d];
                                temp_inner /= self->shape[d];
                            }
                            size_t temp_outer = outer;
                            for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
                                back_coords[d] = temp_outer % self->shape[d];
                                temp_outer /= self->shape[d];
                            }

                            size_t self_flat = self->get_flat_index(back_coords);
                            (*self->grad)[self_flat] += upstream_grad;
                        }
                    }
                }
            }
        }
    };

    return out;
}

TensorPtr Tensor::transpose(size_t dim0, size_t dim1) {
    if (dim0 >= shape.size() || dim1 >= shape.size()) {
        throw std::invalid_argument("transpose dimensions out of bounds");
    }

    std::vector<size_t> new_shape = shape;
    std::vector<size_t> new_strides = strides;

    // swap metadata dimensions
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    if(this->requires_grad == true){
        this->ensure_grad_allocated();
    }

    // build the new output tensor
    auto out = std::make_shared<Tensor>(this->data, this->grad, new_shape, std::vector<TensorPtr>{shared_from_this()}, "transpose");
    out->strides = new_strides; // override default contigous layout strides

    return out;
}



// -------------------------------
// unary operators
// -------------------------------

// ReLU
TensorPtr Tensor::relu(){
    // enforce contiguity
    auto active_this = is_contiguous() ? shared_from_this() : contiguous();

    size_t total_elements = 1;
    for (size_t s : active_this->shape) total_elements *= s;

    std::vector<double> dummy_vals(total_elements);
    auto out = std::make_shared<Tensor>(std::move(dummy_vals), active_this->shape, std::vector<TensorPtr>{active_this}, "relu");
    out->to(active_this->device);

    // ---- forward pass
    if(active_this->device == Device::CUDA){
        // --- gpu forward pass
        launch_unary_forward(
            active_this->cuda_data, 
            out->cuda_data, 
            total_elements, 
            reluForwardOp()
        );

    } else {
        // --- cpu forward pass
        double* out_ptr = out->data->data();
        double* in_ptr = active_this->data->data();

        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            out_ptr[i] = in_ptr[i] > 0.0 ? in_ptr[i] : 0.0;
        }
    }

    // ----- backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = active_this;

    out->backward_func = [self, weak_out, total_elements](){
        if(auto out_ptr = weak_out.lock()){
            if(self->requires_grad) {

                // GPU backward pass
                if(self->device == Device::CUDA){
                    launch_unary_backward(
                        out_ptr->cuda_grad, 
                        self->cuda_grad, 
                        self->cuda_data, 
                        total_elements, 
                        reluBackwardOp()
                    );

                } else {
                    // CPU backward pass
                    for(size_t i = 0; i < total_elements; ++i){
                        double local_derivative = (*self->data)[i] > 0.0 ? 1.0 : 0.0;
                        (*self->grad)[i] += (*out_ptr->grad)[i] * local_derivative;
                    }
                }
                
            }
        }
    };

    return out;
}

// exp function
TensorPtr Tensor::exp(){
    auto active_this = is_contiguous() ? shared_from_this() : contiguous();

    size_t total_elements = 1;
    for (size_t s : active_this->shape) total_elements *= s;

    std::vector<double> dummy_vals(total_elements);
    auto out = std::make_shared<Tensor>(std::move(dummy_vals), active_this->shape, std::vector<TensorPtr>{active_this}, "exp");
    out->to(active_this->device);

    // --- forward pass
    if(active_this->device == Device::CUDA){
        // --- gpu forward pass
        launch_unary_forward(
            active_this->cuda_data, 
            out->cuda_data, 
            total_elements, 
            expForwardOp()
        );

    } else {
        // -- cpu forward pass
        double* out_ptr = out->data->data();
        double* in_ptr = active_this->data->data();
        
        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            out_ptr[i] = std::exp(in_ptr[i]);
        }
    }


    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = active_this;

    out->backward_func = [self, weak_out, total_elements]() {
        if(auto out_ptr = weak_out.lock()){
            if (self->requires_grad) {

                // -- gpu
                if(self->device == Device::CUDA){
                    launch_unary_backward(
                        out_ptr->cuda_grad, 
                        self->cuda_grad, 
                        self->cuda_data, 
                        total_elements, 
                        expBackwardOp()
                    );

                } else {
                    // --- cpu
                    for(size_t i = 0; i < total_elements; ++i){
                        double local_derivative = std::exp((*self->data)[i]);
                        (*self->grad)[i] += (*out_ptr->grad)[i] * local_derivative;
                    }
                }

            }

        }
    };

    return out;
}

//tanh function
TensorPtr Tensor::tanh(){
    auto active_this = is_contiguous() ? shared_from_this() : contiguous();

    size_t total_elements = 1;
    for (size_t s : active_this->shape) total_elements *= s;

    std::vector<double> dummy_vals(total_elements);
    auto out = std::make_shared<Tensor>(std::move(dummy_vals), active_this->shape, std::vector<TensorPtr>{active_this}, "tanh");
    out->to(active_this->device);

    // --- forward pass
    if(active_this->device == Device::CUDA){
        // --- gpu
        launch_unary_forward(
            active_this->cuda_data, 
            out->cuda_data, 
            total_elements, 
            tanhForwardOp()
        );

    } else {
        // --cpu
        double* out_ptr = out->data->data();
        double* in_ptr = active_this->data->data();

        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            out_ptr[i] = std::tanh(in_ptr[i]);
        }
    }


    // --- backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = active_this;

    out->backward_func = [self, weak_out, total_elements](){
        if(auto out_ptr = weak_out.lock()){
            if(self->requires_grad){

                // gpu backward
                if(self->device == Device::CUDA){
                    launch_unary_backward(
                        out_ptr->cuda_grad, 
                        self->cuda_grad, 
                        self->cuda_data, 
                        total_elements, 
                        tanhBackwardOp()
                    );

                } else {
                    // --- cpu
                    for(size_t i = 0; i < total_elements; ++i){
                        double t = std::tanh((*self->data)[i]);
                        double local_derivative = 1.0 - t * t;
                        (*self->grad)[i] += (*out_ptr->grad)[i] * local_derivative;
                    }
                }

            }
        }
    };

    return out;
}


// sigmoid function
TensorPtr Tensor::sigmoid(){
    auto active_this = is_contiguous() ? shared_from_this() : contiguous();

    size_t total_elements = 1;
    for (size_t s : shape) total_elements *= s;

    std::vector<double> dummy_vals(total_elements);
    auto out = std::make_shared<Tensor>(std::move(dummy_vals), active_this->shape, std::vector<TensorPtr>{active_this}, "sigmoid");
    out->to(active_this->device);
    
    // forward pass
    if(active_this->device == Device::CUDA){
        // --- gpu forward pass
        launch_unary_forward(
            active_this->cuda_data, 
            out->cuda_data, 
            total_elements, 
            sigmoidForwardOp()
        );

    } else {
        // --- cpu
        double* out_ptr = out->data->data();
        double* in_ptr = active_this->data->data();

        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            out_ptr[i] = 1.0 / (1.0 + std::exp(-in_ptr[i]));
        }
    }


    //backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = active_this;

    out->backward_func = [self, weak_out, total_elements]() {
        if(auto out_ptr = weak_out.lock()){
            if(self->requires_grad){

                // GPU backward pass
                if(self->device == Device::CUDA){
                    launch_unary_backward(
                        out_ptr->cuda_grad, 
                        self->cuda_grad, 
                        self->cuda_data, 
                        total_elements, 
                        sigmoidBackwardOp()
                    );

                } else {
                    // CPU backward pass
                    for(size_t i = 0; i < total_elements; ++i){
                        double s = (*self->data)[i];
                        (*self->grad)[i] += (*out_ptr->grad)[i] * (s * (1.0 - s));
                    }
                }

            }
        }
    };

    return out;
}

// log function (natural log)
TensorPtr Tensor::log(){
    auto active_this = is_contiguous() ? shared_from_this() : contiguous();

    size_t total_elements = 1;
    for (size_t s : active_this->shape) total_elements *= s;

    std::vector<double> dummy_vals(total_elements);
    auto out = std::make_shared<Tensor>(std::move(dummy_vals), active_this->shape, std::vector<TensorPtr>{active_this}, "log");
    out->to(active_this->device);
    
    // forward pass
    if(active_this->device == Device::CUDA){
        // --- gpu forward pass
        launch_unary_forward(
            active_this->cuda_data,
            out->cuda_data,
            total_elements,
            logForwardOp()
        );

    } else {
        // ---- cpu
        double* out_ptr = out->data->data();
        double* in_ptr = active_this->data->data();

        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            out_ptr[i] = std::log(in_ptr[i]);
        }
    }


    // backward pass
    auto self = active_this;
    std::weak_ptr<Tensor> weak_out = out;

    out->backward_func = [self, weak_out, total_elements]() {
        if(auto out_ptr = weak_out.lock()){
            if(self->requires_grad){

                // GPU backward pass
                if(self->device == Device::CUDA){
                    launch_unary_backward(
                        out_ptr->cuda_grad,
                        self->cuda_grad,
                        self->cuda_data,
                        total_elements,
                        logBackwardOp()
                    );

                } else{
                    //cpu backward
                    std::vector<size_t> back_coords(self->shape.size(), 0);
                    for(size_t i = 0; i < total_elements; ++i){
                        (*self->grad)[i] += (*out_ptr->grad)[i] * (1.0 / (*self->data)[i]);
                    }
                }

            }
        }
    };

    return out;
}

// pow function
TensorPtr Tensor::pow(double exponent){
    auto active_this = is_contiguous() ? shared_from_this() : contiguous();

    size_t total_elements = 1;
    for (size_t s : active_this->shape) total_elements *= s;

    std::vector<double> dummy_vals(total_elements);
    auto out = std::make_shared<Tensor>(std::move(dummy_vals), active_this->shape, std::vector<TensorPtr>{active_this}, "pow");
    out->to(active_this->device);

    // forward pass
    if(active_this->device == Device::CUDA){
        // gpu forward
        launch_unary_forward(
            active_this->cuda_data,
            out->cuda_data,
            total_elements,
            powForwardOp(exponent)
        );

    } else {
        // forward on cpu
        double* out_ptr = out->data->data();
        double* in_ptr = active_this->data->data();

        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            out_ptr[i] = std::pow(in_ptr[i], exponent);
        }
    }


    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = active_this;

    // explicitly capture the exponent by value
    out->backward_func = [self, weak_out, exponent, total_elements]() {
        if (auto out_ptr = weak_out.lock()) {
            if(self->requires_grad){

                // gpu backward pass
                if(self->device == Device::CUDA){
                    launch_unary_backward(
                        out_ptr->cuda_grad,
                        self->cuda_grad,
                        self->cuda_data,
                        total_elements,
                        powBackwardOp(exponent)
                    );

                } else {
                    // cpu backward pass
                    for (size_t i = 0; i < total_elements; ++i) {
                        double local_derivative = exponent * std::pow((*self->data)[i], exponent - 1.0);
                        (*self->grad)[i] += (*out_ptr->grad)[i] * local_derivative;
                    }
                }
            }
        }
    };

    return out;
}

TensorPtr Tensor::sqrt(){
    auto active_this = is_contiguous() ? shared_from_this() : contiguous();

    size_t total_elements = 1;
    for (size_t s : active_this->shape) total_elements *= s;

    std::vector<double> dummy_vals(total_elements);
    auto out = std::make_shared<Tensor>(std::move(dummy_vals), active_this->shape, std::vector<TensorPtr>{active_this}, "sqrt");
    out->to(active_this->device);
    
    // forward pass
    if(active_this->device == Device::CUDA){
        // --- gpu forward pass
        launch_unary_forward(
            active_this->cuda_data,
            out->cuda_data,
            total_elements,
            sqrtForwardOp()
        );

    } else {
        // ---- cpu forward
        double* out_ptr = out->data->data();
        double* in_ptr = active_this->data->data();

        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            if (in_ptr[i] < 0.0) {
                throw std::runtime_error("sqrt: cannot calculate square root of a negative value");
            }
            out_ptr[i] = std::sqrt(in_ptr[i]);
        }
    }


    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();

    out->backward_func = [self, weak_out, total_elements]() {
        if (auto out_ptr = weak_out.lock()) {
            if (self->requires_grad) {

                if(self->device == Device::CUDA){
                    launch_unary_backward(
                        out_ptr->cuda_grad,
                        self->cuda_grad,
                        self->cuda_data,
                        total_elements,
                        sqrtBackwardOp()
                    );

                } else {
                    // backward pass on cpu
                    for (size_t i = 0; i < out_ptr->data->size(); ++i) {
                        // derivative of sqrt(x) is 0.5 / sqrt(x)
                        // guard against 0.0 inputs to avoid inf expansion and nan graph corruption
                        double local_derivative = ((*out_ptr->data)[i] == 0.0) ? 0.0 : (0.5 / (*out_ptr->data)[i]);
                        (*self->grad)[i] += (*out_ptr->grad)[i] * local_derivative;
                    }
                }

            }
        }
    };

    return out;
}

// element-wise negation
TensorPtr Tensor::neg(){    
    auto active_this = is_contiguous() ? shared_from_this() : contiguous();

    size_t total_elements = 1;
    for (size_t s : shape) total_elements *= s;

    std::vector<double> dummy_vals(total_elements);
    auto out = std::make_shared<Tensor>(std::move(dummy_vals), active_this->shape, std::vector<TensorPtr>{active_this}, "neg");
    out->to(active_this->device);
    
    // forward pass
    if(active_this->device == Device::CUDA){
        // --- gpu
        launch_unary_forward(
            active_this->cuda_data,
            out->cuda_data,
            total_elements,
            negForwardOp()
        );

    } else {
        // cpu
        double* out_ptr = out->data->data();
        double* in_ptr = active_this->data->data();

        // stateless coordinate resolution for safe parallelization
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(total_elements); ++i) {
            out_ptr[i] = -in_ptr[i];
        }
    }


    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = active_this;

    out->backward_func = [self, weak_out, total_elements]() {
        if (auto out_ptr = weak_out.lock()) {
            if (self->requires_grad) {

                // --- gpu
                if(self->device == Device::CUDA){
                    launch_unary_backward(
                        out_ptr->cuda_grad,
                        self->cuda_grad,
                        self->cuda_data,
                        total_elements,
                        negBackwardOp()
                    );

                } else {
                    // cpu
                    for (size_t i = 0; i < out_ptr->data->size(); ++i) {
                        // derivative of -x is -1.0
                        (*self->grad)[i] -= (*out_ptr->grad)[i];
                    }
                }

            }
        }
    };

    return out;
}

// unary prefix negation operator routing
TensorPtr operator-(const TensorPtr& tensor) {
    return tensor->neg();
}

// ------------------------------------------



// softmax
TensorPtr Tensor::softmax(size_t dim) {
    // find max values along target dimension with keepdim=true for broadcasting
    auto max_val = max(dim, true);
    
    // subtract the max to shift logits and prevent exponential overflow explosion
    auto shifted = shared_from_this() - max_val;
    
    // compute numerators
    auto exps = shifted->exp();
    
    // compute denominators along the same target dimension axis block
    auto sum_exps = exps->sum(dim, true);
    
    // element-wise division computes final probabilities seamlessly via broadcasting
    return exps / sum_exps;
}

// log_softmax using the log-sum-exp trick
TensorPtr Tensor::log_softmax(size_t dim) {
    auto max_val = max(dim, true);
    auto shifted = shared_from_this() - max_val;
    auto exps = shifted->exp();
    auto sum_exps = exps->sum(dim, true);
    
    // x - x_max - log(sum(exp(x - x_max)))
    return shifted - sum_exps->log();
}

// tensor->mean() function
TensorPtr Tensor::mean(size_t dim, bool keepdim){
    ReductionMeta meta = prepare_reduction_metadata(dim, keepdim);
    
    std::vector<double> out_vals(meta.total_out_elements, 0.0);

    // forward pass
    for(size_t outer {0}; outer < meta.outer_block_size; ++outer){
        for(size_t inner {0}; inner < meta.inner_block_size; ++inner){
            size_t out_flat = outer * meta.inner_block_size + inner;
            
            double sum = 0.0;

            for(size_t r {0}; r < meta.reduced_size; ++r){
                std::vector<size_t> coords(shape.size());
                coords[dim] = r;

                size_t temp_inner = inner;
                for (size_t d = shape.size() - 1; d > dim; --d) {
                    coords[d] = temp_inner % shape[d];
                    temp_inner /= shape[d];
                }
                size_t temp_outer = outer;
                for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
                    coords[d] = temp_outer % shape[d];
                    temp_outer /= shape[d];
                }

                size_t self_flat = get_flat_index(coords);
                sum += (*data)[self_flat];
            }

            out_vals[out_flat] = sum / meta.reduced_size;
        }
    }

    auto out = std::make_shared<Tensor>(out_vals, meta.out_shape, std::vector<TensorPtr>{shared_from_this()}, "mean");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();

    size_t outer_bs = meta.outer_block_size;
    size_t inner_bs = meta.inner_block_size;
    size_t r_size = meta.reduced_size;

    out->backward_func = [self, weak_out, outer_bs, inner_bs, r_size, dim](){
        if(auto out_ptr = weak_out.lock()){
            // derivative of a mean is just 1/N
            double grad_scale = 1.0 / static_cast<double>(r_size);

            for (size_t outer {0}; outer < outer_bs; ++outer) {
                for (size_t inner {0}; inner < inner_bs; ++inner) {
                    size_t out_flat = outer * inner_bs + inner;
                    
                    // grab the upstream gradient for this specific block
                    double upstream_grad = (*out_ptr->grad)[out_flat] * grad_scale;

                    // distribute it equally to all elements that formed the mean
                    if(self->requires_grad){
                        for (size_t r {0}; r < r_size; ++r) {
                            std::vector<size_t> back_coords(self->shape.size());
                            back_coords[dim] = r;
                            
                            size_t temp_inner = inner;
                            for (size_t d = self->shape.size() - 1; d > dim; --d) {
                                back_coords[d] = temp_inner % self->shape[d];
                                temp_inner /= self->shape[d];
                            }
                            size_t temp_outer = outer;
                            for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
                                back_coords[d] = temp_outer % self->shape[d];
                                temp_outer /= self->shape[d];
                            }

                            size_t self_flat = self->get_flat_index(back_coords);
                            (*self->grad)[self_flat] += upstream_grad;
                        }
                    }
                }
            }
        }
    };

    return out;
}

// tensor->max() function
TensorPtr Tensor::max(size_t dim, bool keepdim){
    ReductionMeta meta = prepare_reduction_metadata(dim, keepdim);
    std::vector<double> out_vals(meta.total_out_elements, 0.0);
    auto max_indices = std::make_shared<std::vector<size_t>>(meta.total_out_elements, 0); // required for backward pass

    // forward pass
    for(size_t outer {0}; outer < meta.outer_block_size; ++outer){
        for(size_t inner {0}; inner < meta.inner_block_size; ++inner){
            size_t out_flat = outer * meta.inner_block_size + inner;

            double current_max = -INFINITY;
            size_t best_flat_idx = 0;

            for(size_t r {0}; r < meta.reduced_size; ++r){
                std::vector<size_t> coords(shape.size());
                coords[dim] = r;
                
                size_t temp_inner = inner;
                for (size_t d = shape.size() - 1; d > dim; --d) {
                    coords[d] = temp_inner % shape[d];
                    temp_inner /= shape[d];
                }
                size_t temp_outer = outer;
                for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
                    coords[d] = temp_outer % shape[d];
                    temp_outer /= shape[d];
                }

                size_t self_flat = get_flat_index(coords);
                if ((*data)[self_flat] > current_max) {
                    current_max = (*data)[self_flat];
                    best_flat_idx = self_flat;
                }
            }
            out_vals[out_flat] = current_max;
            (*max_indices)[out_flat] = best_flat_idx;
        }
    }

    auto out = std::make_shared<Tensor>(out_vals, meta.out_shape, std::vector<TensorPtr>{shared_from_this()}, "max");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();
    size_t total_out_elements = meta.total_out_elements;

    out->backward_func = [self, weak_out, max_indices, total_out_elements]() {
        if(auto out_ptr = weak_out.lock()){
            // gradient routing bypasses block math entirely; just map flat index to flat index
            if(self->requires_grad){
                for (size_t i {0}; i < total_out_elements; ++i) {
                    size_t winner_flat_idx = (*max_indices)[i];
                    (*self->grad)[winner_flat_idx] += (*out_ptr->grad)[i];
                }
            }
        }
    };

    return out;
}

// tensor->min() function
TensorPtr Tensor::min(size_t dim, bool keepdim){
    ReductionMeta meta = prepare_reduction_metadata(dim, keepdim);

    std::vector<double> out_vals(meta.total_out_elements, 0.0);

    auto min_indices = std::make_shared<std::vector<size_t>>(meta.total_out_elements, 0); // required for backward pass

    // forward pass
    for(size_t outer {0}; outer < meta.outer_block_size; ++outer){
        for(size_t inner {0}; inner < meta.inner_block_size; ++inner){
            size_t out_flat = outer * meta.inner_block_size + inner;

            double current_min = INFINITY;
            size_t best_flat_idx = 0;

            for(size_t r {0}; r < meta.reduced_size; ++r){
                std::vector<size_t> coords(shape.size());
                coords[dim] = r;
                
                size_t temp_inner = inner;
                for (size_t d = shape.size() - 1; d > dim; --d) {
                    coords[d] = temp_inner % shape[d];
                    temp_inner /= shape[d];
                }
                size_t temp_outer = outer;
                for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
                    coords[d] = temp_outer % shape[d];
                    temp_outer /= shape[d];
                }

                size_t self_flat = get_flat_index(coords);
                if ((*data)[self_flat] < current_min) {
                    current_min = (*data)[self_flat];
                    best_flat_idx = self_flat;
                }
            }
            out_vals[out_flat] = current_min;
            (*min_indices)[out_flat] = best_flat_idx;
        }
    }

    auto out = std::make_shared<Tensor>(out_vals, meta.out_shape, std::vector<TensorPtr>{shared_from_this()}, "min");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();
    size_t total_out_elements = meta.total_out_elements;

    out->backward_func = [self, weak_out, min_indices, total_out_elements]() {
        if(auto out_ptr = weak_out.lock()){
            // gradient routing bypasses block math entirely; just map flat index to flat index
            if(self->requires_grad){
                for (size_t i {0}; i < total_out_elements; ++i) {
                    size_t winner_flat_idx = (*min_indices)[i];
                    (*self->grad)[winner_flat_idx] += (*out_ptr->grad)[i];
                }
            }
        }
    };

    return out;
}

// tensor->argmax() function
TensorPtr Tensor::argmax(size_t dim, bool keepdim) {
    ReductionMeta meta = prepare_reduction_metadata(dim, keepdim);
    std::vector<double> out_vals(meta.total_out_elements, 0.0);

    // forward pass
    for (size_t outer {0}; outer < meta.outer_block_size; ++outer) {
        for (size_t inner {0}; inner < meta.inner_block_size; ++inner) {
            size_t out_flat = outer * meta.inner_block_size + inner;
            double current_max = -INFINITY;
            size_t best_r = 0; // Track the relative index 'r'

            for (size_t r {0}; r < meta.reduced_size; ++r) {
                std::vector<size_t> coords(shape.size());
                coords[dim] = r;
                
                size_t temp_inner = inner;
                for (size_t d = shape.size() - 1; d > dim; --d) {
                    coords[d] = temp_inner % shape[d];
                    temp_inner /= shape[d];
                }
                size_t temp_outer = outer;
                for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
                    coords[d] = temp_outer % shape[d];
                    temp_outer /= shape[d];
                }

                size_t self_flat = get_flat_index(coords);
                if ((*data)[self_flat] > current_max) {
                    current_max = (*data)[self_flat];
                    best_r = r;
                }
            }
            // Output the index (cast to double), not the value
            out_vals[out_flat] = static_cast<double>(best_r);
        }
    }

    auto out = std::make_shared<Tensor>(out_vals, meta.out_shape, std::vector<TensorPtr>{}, "argmax");
    
    out->requires_grad = false;
    
    return out;
}

// tensor->argmin() function
TensorPtr Tensor::argmin(size_t dim, bool keepdim) {
    ReductionMeta meta = prepare_reduction_metadata(dim, keepdim);

    std::vector<double> out_vals(meta.total_out_elements, 0.0);

    // forward pass
    for (size_t outer {0}; outer < meta.outer_block_size; ++outer) {
        for (size_t inner {0}; inner < meta.inner_block_size; ++inner) {
            size_t out_flat = outer * meta.inner_block_size + inner;
            double current_min = INFINITY;
            size_t best_r = 0; // Track the relative index 'r'

            for (size_t r {0}; r < meta.reduced_size; ++r) {
                std::vector<size_t> coords(shape.size());
                coords[dim] = r;
                
                size_t temp_inner = inner;
                for (size_t d = shape.size() - 1; d > dim; --d) {
                    coords[d] = temp_inner % shape[d];
                    temp_inner /= shape[d];
                }
                size_t temp_outer = outer;
                for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
                    coords[d] = temp_outer % shape[d];
                    temp_outer /= shape[d];
                }

                size_t self_flat = get_flat_index(coords);
                if ((*data)[self_flat] < current_min) {
                    current_min = (*data)[self_flat];
                    best_r = r;
                }
            }
            // Output the index (cast to double), not the value
            out_vals[out_flat] = static_cast<double>(best_r);
        }
    }

    auto out = std::make_shared<Tensor>(out_vals, meta.out_shape, std::vector<TensorPtr>{}, "argmin");
    
    out->requires_grad = false;
    
    return out;
}

// tensor.reshape(), reshapes tensor dims while preserving element layout sequence
TensorPtr Tensor::reshape(const std::vector<size_t>& new_shape){
    // force contiguity if layout is non-contiguous
    auto active_this = is_contiguous() ? shared_from_this() : contiguous();

    size_t old_elements = 1;
    for (size_t s : active_this->shape) old_elements *= s;

    size_t new_elements = 1;
    for (size_t s: new_shape) new_elements *= s;

    if(old_elements != new_elements){
        throw std::invalid_argument("reshape: total number of elements must match original shape capacity");
    }

    // compute standard contiguous row-major strides for the new shape
    std::vector<size_t> new_strides(new_shape.size(), 1);
    if (!new_shape.empty()) {
        for (int i = static_cast<int>(new_shape.size()) - 2; i >= 0; --i) {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }
    }

    if (active_this->requires_grad) {
        active_this->ensure_grad_allocated();
    }

    auto out = std::make_shared<Tensor>(active_this->data, active_this->grad, new_shape, std::vector<TensorPtr>{active_this}, "reshape");
    out->strides = new_strides;
    return out;
}

// tensor.squeeze(), drops a dimension of size 1 at the specified index axis
TensorPtr Tensor::squeeze(size_t dim){
    if (dim >= shape.size()) {
        throw std::out_of_range("squeeze: dimension index out of bounds");
    }

    // if the dimension is not 1, squeezing acts as a safe no-op view
    if (shape[dim] != 1) {
        return shared_from_this();
    }

    std::vector<size_t> new_shape = shape;
    std::vector<size_t> new_strides = strides;

    new_shape.erase(new_shape.begin() + dim);
    new_strides.erase(new_strides.begin() + dim);

    if (this->requires_grad) {
        this->ensure_grad_allocated();
    }

    auto out = std::make_shared<Tensor>(this->data, this->grad, new_shape, std::vector<TensorPtr>{shared_from_this()}, "squeeze");
    out->strides = new_strides;
    return out;
}

// tensor.unsqueeze(), inserts a new dimension of size 1 at the specified index axis
TensorPtr Tensor::unsqueeze(size_t dim) {
    if (dim > shape.size()) {
        throw std::out_of_range("unsqueeze: dimension index out of bounds");
    }

    std::vector<size_t> new_shape = shape;
    std::vector<size_t> new_strides = strides;

    new_shape.insert(new_shape.begin() + dim, 1);
    
    // inject layout-safe stride alignment matching the adjacent memory block jumps
    if (dim >= strides.size()) {
        new_strides.push_back(1);
    } else {
        new_strides.insert(new_strides.begin() + dim, strides[dim] * shape[dim]);
    }

    if (this->requires_grad) {
        this->ensure_grad_allocated();
    }

    auto out = std::make_shared<Tensor>(this->data, this->grad, new_shape, std::vector<TensorPtr>{shared_from_this()}, "unsqueeze");
    out->strides = new_strides;
    return out;
}

// tensor.permute(), reorders layout dimensions based on an arbitrary axis permutation sequence
TensorPtr Tensor::permute(const std::vector<size_t>& dims) {
    if (dims.size() != shape.size()) {
        throw std::invalid_argument("permute: permutation vector must match tensor rank size");
    }

    // validate completeness of dimension indexes to prevent axis duplication/omission
    std::vector<bool> seen(shape.size(), false);
    for (size_t d : dims) {
        if (d >= shape.size()) {
            throw std::out_of_range("permute: axis index out of bounds");
        }
        if (seen[d]) {
            throw std::invalid_argument("permute: duplicate axis entries detected");
        }
        seen[d] = true;
    }

    std::vector<size_t> new_shape(shape.size());
    std::vector<size_t> new_strides(shape.size());

    // map the shape and stride layouts directly onto the new target positions
    for (size_t i = 0; i < dims.size(); ++i) {
        new_shape[i] = shape[dims[i]];
        new_strides[i] = strides[dims[i]];
    }

    if (this->requires_grad) {
        this->ensure_grad_allocated();
    }

    // permute creates a non-contiguous structural view node
    auto out = std::make_shared<Tensor>(this->data, this->grad, new_shape, std::vector<TensorPtr>{shared_from_this()}, "permute");
    out->strides = new_strides;
    return out;
}

// equals to comparison
TensorPtr operator==(const Tensor& lhs, const Tensor& rhs) {
    std::vector<size_t> out_shape;
    std::vector<size_t> lhs_b_strides;
    std::vector<size_t> rhs_b_strides;
    
    // leverage zero-copy layout view constructors instead of cloning memory vectors
    auto lhs_ptr = std::make_shared<Tensor>(lhs.data, lhs.grad, lhs.shape, std::vector<TensorPtr>{}, "");
    auto rhs_ptr = std::make_shared<Tensor>(rhs.data, rhs.grad, rhs.shape, std::vector<TensorPtr>{}, "");
    lhs_ptr->strides = lhs.strides;
    rhs_ptr->strides = rhs.strides;
    
    Tensor::compute_broadcast_metadata(lhs_ptr, rhs_ptr, out_shape, lhs_b_strides, rhs_b_strides);

    size_t total_elements = 1;
    for (size_t dim : out_shape) total_elements *= dim;
    std::vector<double> out_values(total_elements);

    // stateless coordinate resolution for safe parallelization
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(total_elements); ++i) {
        size_t flat_lhs = 0;
        size_t flat_rhs = 0;
        size_t temp = i;
        for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
            size_t coord = temp % out_shape[d];
            flat_lhs += coord * lhs_b_strides[d];
            flat_rhs += coord * rhs_b_strides[d];
            temp /= out_shape[d];
        }
        out_values[i] = ((*lhs.data)[flat_lhs] == (*rhs.data)[flat_rhs]) ? 1.0 : 0.0;
    }

    return std::make_shared<Tensor>(out_values, out_shape, std::vector<TensorPtr>{}, "==");
}

// less than comparison
TensorPtr operator<(const Tensor& lhs, const Tensor& rhs) {
    std::vector<size_t> out_shape;
    std::vector<size_t> lhs_b_strides;
    std::vector<size_t> rhs_b_strides;
    
    auto lhs_ptr = std::make_shared<Tensor>(lhs.data, lhs.grad, lhs.shape, std::vector<TensorPtr>{}, "");
    auto rhs_ptr = std::make_shared<Tensor>(rhs.data, rhs.grad, rhs.shape, std::vector<TensorPtr>{}, "");
    lhs_ptr->strides = lhs.strides;
    rhs_ptr->strides = rhs.strides;
    
    Tensor::compute_broadcast_metadata(lhs_ptr, rhs_ptr, out_shape, lhs_b_strides, rhs_b_strides);

    size_t total_elements = 1;
    for (size_t dim : out_shape) total_elements *= dim;
    std::vector<double> out_values(total_elements);

    // stateless coordinate resolution for safe parallelization
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(total_elements); ++i) {
        size_t flat_lhs = 0;
        size_t flat_rhs = 0;
        size_t temp = i;
        for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
            size_t coord = temp % out_shape[d];
            flat_lhs += coord * lhs_b_strides[d];
            flat_rhs += coord * rhs_b_strides[d];
            temp /= out_shape[d];
        }
        out_values[i] = ((*lhs.data)[flat_lhs] < (*rhs.data)[flat_rhs]) ? 1.0 : 0.0;
    }

    return std::make_shared<Tensor>(out_values, out_shape, std::vector<TensorPtr>{}, "<");
}

// greater than comparison
TensorPtr operator>(const Tensor& lhs, const Tensor& rhs) {
    std::vector<size_t> out_shape;
    std::vector<size_t> lhs_b_strides;
    std::vector<size_t> rhs_b_strides;
    
    auto lhs_ptr = std::make_shared<Tensor>(lhs.data, lhs.grad, lhs.shape, std::vector<TensorPtr>{}, "");
    auto rhs_ptr = std::make_shared<Tensor>(rhs.data, rhs.grad, rhs.shape, std::vector<TensorPtr>{}, "");
    lhs_ptr->strides = lhs.strides;
    rhs_ptr->strides = rhs.strides;
    
    Tensor::compute_broadcast_metadata(lhs_ptr, rhs_ptr, out_shape, lhs_b_strides, rhs_b_strides);

    size_t total_elements = 1;
    for (size_t dim : out_shape) total_elements *= dim;
    std::vector<double> out_values(total_elements);

    // stateless coordinate resolution for safe parallelization
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(total_elements); ++i) {
        size_t flat_lhs = 0;
        size_t flat_rhs = 0;
        size_t temp = i;
        for (int d = static_cast<int>(out_shape.size()) - 1; d >= 0; --d) {
            size_t coord = temp % out_shape[d];
            flat_lhs += coord * lhs_b_strides[d];
            flat_rhs += coord * rhs_b_strides[d];
            temp /= out_shape[d];
        }
        out_values[i] = ((*lhs.data)[flat_lhs] > (*rhs.data)[flat_rhs]) ? 1.0 : 0.0;
    }

    return std::make_shared<Tensor>(out_values, out_shape, std::vector<TensorPtr>{}, ">");
}

// explicitly expands tensor dimensions along singleton axes via zero-copy stride mapping
TensorPtr Tensor::expand(const std::vector<size_t>& new_shape){
    size_t old_ndim = shape.size();
    size_t new_ndim = new_shape.size();

    // validate rank boundary condition
    if (new_ndim < old_ndim){
        throw std::invalid_argument("expand: requested shape rank cannot be smaller than current tensor rank");
    }

    // adapt and left-pad the original shape/strides to match the target rank configuration
    std::vector<size_t> adjusted_old_shape(new_ndim, 1);
    std::vector<size_t> adjusted_old_strides(new_ndim, 0);
    for (size_t i = 0; i < old_ndim; ++i) {
        adjusted_old_shape[new_ndim - 1 - i] = shape[old_ndim - 1 - i];
        adjusted_old_strides[new_ndim - 1 - i] = strides[old_ndim - 1 - i];
    }

    std::vector<size_t> new_strides(new_ndim, 0);
    size_t total_new_elements = 1;
    // validate dimension scalability and apply zero-stride assignment
    for (size_t i = 0; i < new_ndim; ++i) {
        total_new_elements *= new_shape[i];
        
        if (adjusted_old_shape[i] == new_shape[i]) {
            new_strides[i] = adjusted_old_strides[i];
        } else if (adjusted_old_shape[i] == 1) {
            new_strides[i] = 0;
        } else {
            throw std::invalid_argument("expand: invalid dimension expansion; can only expand singleton axes of size 1");
        }
    }

    if (this->requires_grad) {
        this->ensure_grad_allocated();
    }

    // allocate a separate gradient vector for the view node to receive full-sized unreduced gradients
    auto new_grad = std::make_shared<std::vector<double>>(total_new_elements, 0.0);

    auto out = std::make_shared<Tensor>(this->data, new_grad, new_shape, std::vector<TensorPtr>{shared_from_this()}, "expand");
    out->strides = new_strides;

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();

    out->backward_func = [self, weak_out, new_shape, total_new_elements]() {
        if (auto out_ptr = weak_out.lock()) {
            if (self->requires_grad) {
                std::vector<size_t> coords(new_shape.size(), 0);
                size_t rank_diff = new_shape.size() - self->shape.size();

                for (size_t i = 0; i < total_new_elements; ++i) {
                    double upstream_grad = (*out_ptr->grad)[i];

                    // compress coordinates back down to the parent's original unexpanded footprint shapes
                    std::vector<size_t> old_coords(self->shape.size());
                    for (size_t d = 0; d < self->shape.size(); ++d) {
                        old_coords[d] = (self->shape[d] == 1) ? 0 : coords[d + rank_diff];
                    }

                    size_t self_flat = self->get_flat_index(old_coords);
                    (*self->grad)[self_flat] += upstream_grad; // accumulates multiple items back to singleton positions

                    Tensor::advance_coordinates(coords, new_shape);
                }
            }
        }
    };

    return out;
}

// argsort along a specific dimension returning sorted index array
TensorPtr Tensor::argsort(size_t dim, bool descending) {
    if (dim >= shape.size()) {
        throw std::invalid_argument("argsort: dimension out of bounds");
    }

    // calculate block sizes for iteration slices
    size_t reduced_size = shape[dim];
    size_t inner_block_size = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        inner_block_size *= shape[i];
    }

    // compute logical elements from shape instead of physical data buffer size
    size_t total_elements = 1;
    for (size_t s : shape) total_elements *= s;
    
    size_t outer_block_size = total_elements / (reduced_size * inner_block_size);

    // allocate contiguous output buffer matching logical size
    std::vector<double> out_vals(total_elements);

    // process each sliced block independently
    for (size_t outer = 0; outer < outer_block_size; ++outer) {
        for (size_t inner = 0; inner < inner_block_size; ++inner) {
            
            // prepare relative index array to sort
            std::vector<size_t> r_indices(reduced_size);
            for (size_t r = 0; r < reduced_size; ++r) r_indices[r] = r;

            // helper to resolve exact non-contiguous source index via coordinate mapping
            auto get_flat_for_r = [&](size_t r) {
                std::vector<size_t> coords(shape.size());
                coords[dim] = r;
                size_t temp_inner = inner;
                for (size_t d = shape.size() - 1; d > dim; --d) {
                    coords[d] = temp_inner % shape[d];
                    temp_inner /= shape[d];
                }
                size_t temp_outer = outer;
                for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
                    coords[d] = temp_outer % shape[d];
                    temp_outer /= shape[d];
                }
                return get_flat_index(coords);
            };

            // sort indices based on underlying tensor values
            std::sort(r_indices.begin(), r_indices.end(), [&](size_t r1, size_t r2) {
                double v1 = (*data)[get_flat_for_r(r1)];
                double v2 = (*data)[get_flat_for_r(r2)];
                return descending ? (v1 > v2) : (v1 < v2);
            });

            // write sorted index array to output vector positions
            for (size_t r = 0; r < reduced_size; ++r) {
                size_t out_flat = outer * (reduced_size * inner_block_size) + r * inner_block_size + inner;
                out_vals[out_flat] = static_cast<double>(r_indices[r]);
            }
        }
    }

    // return non-differentiable tracking node
    auto out = std::make_shared<Tensor>(out_vals, shape, std::vector<TensorPtr>{}, "argsort");
    out->requires_grad = false;
    return out;
}

// backward pass function
void Tensor::backward() {
    // build topological sort list
    std::vector<TensorPtr> topo;
    std::set<TensorPtr> visited;

    std::function<void(TensorPtr)> build_topo = [&](TensorPtr v) {
        if(visited.find(v) == visited.end()){
            visited.insert(v);
            for(const auto& child : v->prev){
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(shared_from_this());
    // allocate gradients for all active graph nodes
    for(auto& node : topo){
        if(node->requires_grad){
            node->ensure_grad_allocated();
        }
    }

    // clear gradients of intermediate nodes for this pass
    std::set<std::vector<double>*> cleared_buffers;

    for(auto& node : topo){
        if(node->requires_grad && !node->prev.empty() && node != shared_from_this()){
            if(node->grad && cleared_buffers.find(node->grad.get()) == cleared_buffers.end()){
                cleared_buffers.insert(node->grad.get());
                std::fill(node->grad->begin(), node->grad->end(), 0.0); // clean pass reset
            }
        }
    }

    // out node start with grad 1.0
    this->ensure_grad_allocated();
    std::fill(grad->begin(), grad->end(), 1.0);

    // process nodes in reverse topo order
    for(auto it = topo.rbegin(); it != topo.rend(); ++it){
        if ((*it)->requires_grad){
            (*it)->backward_func();
        }
    }
}

// tensor.view() function
TensorPtr Tensor::view(const std::vector<int>& target_shape){
    if (!is_contiguous()) {
        throw std::runtime_error("View is incompatible with non-contiguous layouts. Explicitly call .contiguous() first.");
    }

    size_t total_elements = this->data->size();
    size_t product_of_other_dims = 1;
    int negative_one_idx = -1;

    // compute product of explicit dimensions and track the -1 placeholder
    for (size_t i = 0; i < target_shape.size(); ++i){
        if(target_shape[i] == -1){
            if(negative_one_idx != -1){
                throw std::invalid_argument("view shape can only contain single a -1 placeholder axis");
            }
            negative_one_idx = static_cast<int>(i);
        } else if(target_shape[i] <= 0){
            throw std::invalid_argument("tensor dimensions must be positive integers");
        } else {
            product_of_other_dims *= target_shape[i];
        }
    }

    // convert target_shape to a standard unsigned shape vector
    std::vector<size_t> resolved_shape(target_shape.begin(), target_shape.end());

    // infer the -1 dimension size if it was provided
    if(negative_one_idx != -1){
        if(total_elements % product_of_other_dims != 0){
            throw std::invalid_argument("total element capacity mismatch for requested shape layout");
        }
        resolved_shape[negative_one_idx] = total_elements / product_of_other_dims;
    // guard if no -1 is provided
    } else if(product_of_other_dims != total_elements){
            throw std::invalid_argument("requested shape does not match total elements");
    }

    // calculate contigous row-major strides for the newly resolved shape layout
    std::vector<size_t> new_strides(resolved_shape.size(), 1);
    if (!resolved_shape.empty()){
        for(int i {static_cast<int>(resolved_shape.size()) - 2}; i >= 0; --i){
            new_strides[i] = new_strides[i + 1] * resolved_shape[i + 1];
        }
    }

    if(this->requires_grad == true){
        this->ensure_grad_allocated();
    }

    // construct and return the view tracking node sharing the original flat data block
    auto out = std::make_shared<Tensor>(this->data, this->grad, resolved_shape, std::vector<TensorPtr>{shared_from_this()}, "view");
    out->strides = new_strides;

    return out;
}

// print function
void Tensor::print() const {
    if (this->data->empty()) {
        std::cout << "[]\n";
        return;
    }

    // compute logical elements from shape instead of physical data buffer size
    size_t total_elements = 1;
    for (size_t s : shape) total_elements *= s;
    
    size_t ndim = this->shape.size();
    std::vector<size_t> current_idx(ndim, 0);

    // print initial outermost structural open brackets
    for (size_t d = 0; d < ndim; ++d) std::cout << "[";

    for (size_t i = 0; i < total_elements; ++i) {
        // retrieve and format the actual scalar element value
        size_t flat_idx = this->get_flat_index(current_idx);
        std::cout << (*this->data)[flat_idx];

        // track how many dimensions hit their limit simultaneously on this step
        size_t close_brackets_count = 0;
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            if (current_idx[d] == this->shape[d] - 1) {
                close_brackets_count++;
            } else {
                break;
            }
        }

        // advance odometer coordinate map
        bool has_more = Tensor::advance_coordinates(current_idx, this->shape);

        if (has_more) {
            if (close_brackets_count > 0) {
                // we closed an inner dimension row block; wrap with brackets and separate lines
                for (size_t b = 0; b < close_brackets_count; ++b) std::cout << "]";
                std::cout << ",\n";
                
                // pad the next lines with opening alignment spaces matching closed depth
                for (size_t s = 0; s < ndim - close_brackets_count; ++s) std::cout << " ";
                for (size_t b = 0; b < close_brackets_count; ++b) std::cout << "[";
            } else {
                // simple element separation within the same innermost line row
                std::cout << ", ";
            }
        } else {
            // very last item execution step; cap off remaining outermost bracket limits
            for (size_t b = 0; b < ndim; ++b) std::cout << "]";
            std::cout << "\n";
        }
    }
}