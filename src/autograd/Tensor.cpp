#include "Tensor.h"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>

// constructor for leaf nodes
Tensor::Tensor(std::vector<double> values, std::vector<size_t> shape, bool requires_grad)
    : data(std::make_shared<std::vector<double>>(values)),
      grad(nullptr),
      shape(shape), requires_grad(requires_grad), op(""), backward_func([](){}) {
    
    // compute row-major strides
    strides.resize(shape.size(), 1);
    if (!shape.empty()){
        for(int i {static_cast<int>(shape.size()) - 2}; i >= 0; --i){
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
}

// graph constructor for operations
Tensor::Tensor(std::vector<double> values, std::vector<size_t> shape, std::vector<TensorPtr> children, std::string operation)
    : data(std::make_shared<std::vector<double>>(values)), 
      grad(nullptr),
      shape(shape), requires_grad(false), prev(children), op(operation), backward_func([](){}){

    // compute strides for the new shape layout
    strides.resize(shape.size(), 1);
    if (!shape.empty()) {
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    // inherit tracking state from unique parent nodes
    for (const auto& child : children) {
        if (child->requires_grad) {
            this->requires_grad = true;
        }
    }
}

// zero-copy view constructor
Tensor::Tensor(std::shared_ptr<std::vector<double>> shared_data, std::shared_ptr<std::vector<double>> shared_grad, std::vector<size_t> shape, std::vector<TensorPtr> children, std::string operation)
    : data(shared_data), grad(shared_grad), shape(shape), requires_grad(false), prev(children), op(operation), backward_func([](){}) {
    
    strides.resize(shape.size(), 1);
    if (!shape.empty()) {
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    for (const auto& child : children) {
        if (child->requires_grad) this->requires_grad = true;
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

void Tensor::ensure_grad_allocated(){
    if(grad == nullptr){
        grad = (std::make_shared<std::vector<double>>(data->size(), 0.0));
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
    
    std::vector<double> contiguous_values(data->size());
    std::vector<size_t> coords(shape.size(), 0);
    for (size_t i = 0; i < data->size(); ++i) {
        contiguous_values[i] = (*data)[get_flat_index(coords)];
        advance_coordinates(coords, shape);
    }
    
    auto out = std::make_shared<Tensor>(contiguous_values, shape, std::vector<TensorPtr>{shared_from_this()}, "contiguous");
    
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();
    out->backward_func = [self, weak_out]() {
        if (auto out_ptr = weak_out.lock()) {
            std::vector<size_t> back_coords(self->shape.size(), 0);
            for (size_t i = 0; i < self->data->size(); ++i) {
                size_t flat_idx = self->get_flat_index(back_coords);
                (*self->grad)[flat_idx] += (*out_ptr->grad)[i];
                advance_coordinates(back_coords, self->shape);
            }
        }
    };
    return out;
}

// in-place addition supporting standard fast-paths and multi-dimensional coordinate loops
TensorPtr Tensor::add_(const TensorPtr& other) {
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
        
        std::vector<size_t> coords(out_shape.size(), 0);
        size_t total_elements = data->size();
        for (size_t i = 0; i < total_elements; ++i) {
            size_t flat_lhs = get_flat_index_from_broadcast(coords, lhs_b_strides);
            size_t flat_rhs = get_flat_index_from_broadcast(coords, rhs_b_strides);
            (*data)[flat_lhs] += (*other->data)[flat_rhs];
            advance_coordinates(coords, out_shape);
        }
    }
    return shared_from_this();
}

// zeroes the internal gradient vector array
void Tensor::zero_grad() {
    if (grad) {
        std::fill(grad->begin(), grad->end(), 0.0);
    }
}

// addition op
TensorPtr operator+(const TensorPtr& lhs, const TensorPtr& rhs){
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
    std::vector<size_t> current_idx(out_shape.size(), 0);
    for (size_t i = 0; i < total_elements; ++i) {
        size_t flat_lhs = Tensor::get_flat_index_from_broadcast(current_idx, lhs_b_strides);
        size_t flat_rhs = Tensor::get_flat_index_from_broadcast(current_idx, rhs_b_strides);
        
        out_values[i] = (*lhs->data)[flat_lhs] + (*rhs->data)[flat_rhs];
        
        // advance the coordinate system to the next spatial block
        Tensor::advance_coordinates(current_idx, out_shape);
    }

    // construct graph node
    auto out = std::make_shared<Tensor>(out_values, out_shape, std::vector<TensorPtr>{lhs, rhs}, "+");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, out_shape, lhs_b_strides, rhs_b_strides, total_elements]() {
        if(auto out_ptr = weak_out.lock()) {
            std::vector<size_t> back_idx(out_shape.size(), 0);

            for(size_t i {0}; i < total_elements; ++i){
                size_t flat_lhs = Tensor::get_flat_index_from_broadcast(back_idx, lhs_b_strides);
                size_t flat_rhs = Tensor::get_flat_index_from_broadcast(back_idx, rhs_b_strides);
                
                // upstream gradient vector aligns perfectly with total_elements sequence
                auto upstream_grad = (*out_ptr->grad)[i];

                // zero-stride values automatically combine multi-dimensional gradients here
                if(lhs->requires_grad) (*lhs->grad)[flat_lhs] += upstream_grad;
                if(rhs->requires_grad) (*rhs->grad)[flat_rhs] += upstream_grad;

                // step backward coordinate tracking layout forward
                Tensor::advance_coordinates(back_idx, out_shape);
            }
        }
    };

    return out;
}

// substraction op
TensorPtr operator-(const TensorPtr& lhs, const TensorPtr& rhs){
    std::vector<size_t> out_shape;
    std::vector<size_t> lhs_b_strides;
    std::vector<size_t> rhs_b_strides;

    // calculate broadcast shapes and zero-strides properties
    Tensor::compute_broadcast_metadata(lhs, rhs, out_shape, lhs_b_strides, rhs_b_strides);

    // calc total elements needed for out data
    size_t total_elements = 1;
    for(size_t dim : out_shape){
        total_elements *= dim;
    }
    std::vector<double> out_values(total_elements, 0.0);

    // forward pass
    std::vector<size_t> current_idx(out_shape.size(), 0);
    for(size_t i {0}; i < total_elements; ++i){
        size_t flat_lhs = Tensor::get_flat_index_from_broadcast(current_idx, lhs_b_strides);
        size_t flat_rhs = Tensor::get_flat_index_from_broadcast(current_idx, rhs_b_strides);

        out_values[i] = (*lhs->data)[flat_lhs] - (*rhs->data)[flat_rhs];

        // advance the coordinate system to the next spatial block
        Tensor::advance_coordinates(current_idx, out_shape);
    }

    // construct graph node
    auto out = std::make_shared<Tensor>(out_values, out_shape, std::vector<TensorPtr>{lhs,rhs}, "-");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, out_shape, lhs_b_strides, rhs_b_strides, total_elements]() {
        if(auto out_ptr = weak_out.lock()){
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
    std::vector<size_t> current_idx(out_shape.size(), 0);
    for (size_t i = 0; i < total_elements; ++i) {
        size_t flat_lhs = Tensor::get_flat_index_from_broadcast(current_idx, lhs_b_strides);
        size_t flat_rhs = Tensor::get_flat_index_from_broadcast(current_idx, rhs_b_strides);
        
        out_values[i] = (*lhs->data)[flat_lhs] * (*rhs->data)[flat_rhs];
        
        // advance the coordinate system to the next spatial block
        Tensor::advance_coordinates(current_idx, out_shape);
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
    std::vector<size_t> current_idx(out_shape.size(), 0);
    for (size_t i = 0; i < total_elements; ++i) {
        size_t flat_lhs = Tensor::get_flat_index_from_broadcast(current_idx, lhs_b_strides);
        size_t flat_rhs = Tensor::get_flat_index_from_broadcast(current_idx, rhs_b_strides);
        
        // standard element-wise division
        out_values[i] = (*lhs->data)[flat_lhs] / (*rhs->data)[flat_rhs];
        
        Tensor::advance_coordinates(current_idx, out_shape);
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

        size_t dim_lhs = 1;
        size_t stride_lhs = 0;
        if(i < lhs_batch_dims){
            dim_lhs = lhs->shape[lhs_batch_dims - 1 - i];
            stride_lhs = lhs->strides[lhs_batch_dims - 1 - i];
        }

        size_t dim_rhs = 1;
        size_t stride_rhs = 0;
        if (i < rhs_batch_dims) {
            dim_rhs = rhs->shape[rhs_batch_dims - 1 - i];
            stride_rhs = rhs->strides[rhs_batch_dims - 1 - i];
        }

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
    for (size_t dim : batch_shape) {
        total_batches *= dim;
    }

    std::vector<double> out_values(total_batches * M * N, 0.0);

    // extract invariant trailing strides for 2D sub-matrix steps
    size_t lhs_stride_M = lhs->strides[lhs_rank - 2];
    size_t lhs_stride_K = lhs->strides[lhs_rank - 1];
    size_t rhs_stride_K = rhs->strides[rhs_rank - 2];
    size_t rhs_stride_N = rhs->strides[rhs_rank - 1];

    // forward pass
    std::vector<size_t> current_batch_idx(batch_shape.size(), 0);
    for (size_t b = 0; b < total_batches; ++b) {
        size_t batch_lhs_off = Tensor::get_flat_index_from_broadcast(current_batch_idx, lhs_batch_strides);
        size_t batch_rhs_off = Tensor::get_flat_index_from_broadcast(current_batch_idx, rhs_batch_strides);
        size_t batch_out_off = b * M * N;

        // standard 2D matrix multiplication multiplication on the current batch slice
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double sum_val = 0.0;
                for (size_t k = 0; k < K; ++k) {
                    size_t lhs_flat = batch_lhs_off + i * lhs_stride_M + k * lhs_stride_K;
                    size_t rhs_flat = batch_rhs_off + k * rhs_stride_K + j * rhs_stride_N;
                    sum_val += (*lhs->data)[lhs_flat] * (*rhs->data)[rhs_flat];
                }
                out_values[batch_out_off + i * N + j] = sum_val;
            }
        }
        Tensor::advance_coordinates(current_batch_idx, batch_shape);
    }

    auto out = std::make_shared<Tensor>(out_values, out_shape, std::vector<TensorPtr>{lhs, rhs}, "matmul");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, batch_shape, lhs_batch_strides, rhs_batch_strides, total_batches, M, K, N, lhs_stride_M, lhs_stride_K, rhs_stride_K, rhs_stride_N]() {
        if (auto out_ptr = weak_out.lock()) {
            std::vector<size_t> back_batch_idx(batch_shape.size(), 0);

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
    };

    return out;
}

// sum function
TensorPtr Tensor::sum(size_t dim, bool keepdim){
    ReductionMeta meta = prepare_reduction_metadata(dim, keepdim);

    std::vector<double> out_vals(meta.total_out_elements, 0.0);

    // forward pass
    for(size_t outer {0}; outer < meta.outer_block_size; ++outer){
        for(size_t inner {0}; inner < meta.inner_block_size; ++inner){
            size_t out_flat = outer * meta.inner_block_size + inner;

            double sum = 0.0;

            for(size_t r {0}; r < meta.reduced_size; ++r){
                size_t self_flat = outer * (meta.reduced_size * meta.inner_block_size) + r * meta.inner_block_size + inner;

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

    out->backward_func = [self, weak_out, outer_bs, inner_bs, r_size](){
        if(auto out_ptr = weak_out.lock()){

            for (size_t outer {0}; outer < outer_bs; ++outer) {
                for (size_t inner {0}; inner < inner_bs; ++inner) {
                    size_t out_flat = outer * inner_bs + inner;
                    double upstream_grad = (*out_ptr->grad)[out_flat];

                    if(self->requires_grad){
                        for (size_t r {0}; r < r_size; ++r) {
                            size_t self_flat = outer * (r_size * inner_bs) + r * inner_bs + inner;
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

// ReLU
TensorPtr Tensor::relu(){
    std::vector<double> out_vals(data->size());
    std::vector<size_t> coords(shape.size(), 0);

    // forward pass
    for(size_t i {0}; i < data->size(); ++i){
        size_t flat_idx = get_flat_index(coords);
        out_vals[i] = (*data)[flat_idx] > 0.0 ? (*data)[flat_idx] : 0.0;
        advance_coordinates(coords, shape);
    }

    auto out = std::make_shared<Tensor>(out_vals, shape, std::vector<TensorPtr>{shared_from_this()}, "relu");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();
    out->backward_func = [self, weak_out](){
        if(auto out_ptr = weak_out.lock()){
            std::vector<size_t> back_coords(self->shape.size(), 0);
            for(size_t i = 0; i < self->data->size(); ++i){
                size_t flat_idx = self->get_flat_index(back_coords);
                double local_derivative = (*self->data)[flat_idx] > 0.0 ? 1.0 : 0.0;
                if (self->requires_grad){
                    (*self->grad)[flat_idx] += (*out_ptr->grad)[i] * local_derivative;
                }
                advance_coordinates(back_coords, self->shape);
            }
        }
    };

    return out;
}

// exp function
TensorPtr Tensor::exp(){
    std::vector<double> out_vals(data->size());
    std::vector<size_t> coords(shape.size(), 0);

    // forward pass
    for(size_t i {0}; i < data->size(); ++i){
        size_t flat_idx = get_flat_index(coords);
        out_vals[i] = std::exp((*data)[flat_idx]);
        advance_coordinates(coords, shape);
    }

    auto out = std::make_shared<Tensor>(out_vals, shape, std::vector<TensorPtr>{shared_from_this()}, "exp");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();
    out->backward_func = [self, weak_out]() {
        if(auto out_ptr = weak_out.lock()){
            std::vector<size_t> back_coords(self->shape.size(), 0);
            for(size_t i = 0; i < self->data->size(); ++i){
                size_t flat_idx = self->get_flat_index(back_coords);
                if(self->requires_grad){
                    (*self->grad)[flat_idx] += (*out_ptr->grad)[i] * (*out_ptr->data)[i];
                }
                advance_coordinates(back_coords, self->shape);
            }
        }
    };

    return out;
}

//tanh function
TensorPtr Tensor::tanh(){
    std::vector<double> out_vals (data->size());
    std::vector<size_t> coords(shape.size(), 0);

    // forward pass
    for(size_t i {0}; i < data->size(); ++i){
        size_t flat_idx = get_flat_index(coords);
        out_vals[i] = std::tanh((*data)[flat_idx]);
        advance_coordinates(coords, shape);
    }

    auto out = std::make_shared<Tensor>(out_vals, shape, std::vector<TensorPtr>{shared_from_this()}, "tanh");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();
    out->backward_func = [self, weak_out](){
        if(auto out_ptr = weak_out.lock()){
            std::vector<size_t> back_coords(self->shape.size(), 0);
            for(size_t i = 0; i < self->data->size(); ++i){
                size_t flat_idx = self->get_flat_index(back_coords);
                double t = (*out_ptr->data)[i];
                if(self->requires_grad){
                    (*self->grad)[flat_idx] += (*out_ptr->grad)[i] * (1.0 - t * t);
                }
                advance_coordinates(back_coords, self->shape);
            }
        }
    };

    return out;
}


// sigmoid function
TensorPtr Tensor::sigmoid(){
    std::vector<double> out_vals(data->size());
    std::vector<size_t> coords(shape.size(), 0);

    // forward pass
    for(size_t i {0}; i < data->size(); ++i){
        size_t flat_idx = get_flat_index(coords);
        out_vals[i] = 1.0 / (1.0 + std::exp(-(*data)[flat_idx]));
        advance_coordinates(coords, shape);
    }

    auto out = std::make_shared<Tensor>(out_vals, shape, std::vector<TensorPtr>{shared_from_this()}, "sigmoid");

    //backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();
    out->backward_func = [self, weak_out]() {
        if(auto out_ptr = weak_out.lock()){
            std::vector<size_t> back_coords(self->shape.size(), 0);
            for(size_t i = 0; i < self->data->size(); ++i){
                size_t flat_idx = self->get_flat_index(back_coords);
                double s = (*out_ptr->data)[i];
                if(self->requires_grad){
                    (*self->grad)[flat_idx] += (*out_ptr->grad)[i] * (s * (1.0 - s));
                }
                advance_coordinates(back_coords, self->shape);
            }
        }
    };

    return out;
}

// log function (natural log)
TensorPtr Tensor::log(){
    std::vector<double> out_vals(data->size());
    std::vector<size_t> coords(shape.size(), 0);

    // forward pass
    for(size_t i {0}; i < data->size(); ++i){
        size_t flat_idx = get_flat_index(coords);
        out_vals[i] = std::log((*data)[flat_idx]);
        advance_coordinates(coords, shape);
    }

    auto out = std::make_shared<Tensor>(out_vals, shape, std::vector<TensorPtr>{shared_from_this()}, "log");

    // backward pass
    auto self = shared_from_this();
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [self, weak_out]() {
        if(auto out_ptr = weak_out.lock()){
            std::vector<size_t> back_coords(self->shape.size(), 0);
            for(size_t i = 0; i < self->data->size(); ++i){
                size_t flat_idx = self->get_flat_index(back_coords);
                if(self->requires_grad){
                    (*self->grad)[flat_idx] += (*out_ptr->grad)[i] * (1.0 / (*self->data)[flat_idx]);
                }
                advance_coordinates(back_coords, self->shape);
            }
        }
    };

    return out;
}

// pow function
TensorPtr Tensor::pow(double exponent){
    std::vector<double> out_vals (data->size());
    std::vector<size_t> coords(shape.size(), 0);

    // forward pass
    for(size_t i {0}; i < data->size(); ++i){
        size_t flat_idx = get_flat_index(coords);
        out_vals[i] = std::pow((*data)[flat_idx], exponent);
        advance_coordinates(coords, shape);
    }

    auto out = std::make_shared<Tensor>(out_vals, shape, std::vector<TensorPtr>{shared_from_this()}, "pow");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();
    // explicitly capture the exponent by value
    out->backward_func = [self, weak_out, exponent]() {
        if (auto out_ptr = weak_out.lock()) {
            std::vector<size_t> back_coords(self->shape.size(), 0);
            for (size_t i = 0; i < self->data->size(); ++i) {
                size_t flat_idx = self->get_flat_index(back_coords);
                double local_derivative = exponent * std::pow((*self->data)[flat_idx], exponent - 1.0);
                if(self->requires_grad){
                    (*self->grad)[flat_idx] += (*out_ptr->grad)[i] * local_derivative;
                }
                advance_coordinates(back_coords, self->shape);
            }
        }
    };

    return out;
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
                size_t self_flat = outer * (meta.reduced_size * meta.inner_block_size) + r * meta.inner_block_size + inner;

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

    out->backward_func = [self, weak_out, outer_bs, inner_bs, r_size](){
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
                            size_t self_flat = outer * (r_size * inner_bs) + r * inner_bs + inner;
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
                size_t self_flat = outer * (meta.reduced_size * meta.inner_block_size) + r * meta.inner_block_size + inner;

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
                size_t self_flat = outer * (meta.reduced_size * meta.inner_block_size) + r * meta.inner_block_size + inner;

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
                size_t self_flat = outer * (meta.reduced_size * meta.inner_block_size) + r * meta.inner_block_size + inner;

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
                size_t self_flat = outer * (meta.reduced_size * meta.inner_block_size) + r * meta.inner_block_size + inner;

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

    for(auto& node : topo){
        if(node->requires_grad == true){
            node->ensure_grad_allocated();
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

    size_t total_elements = this->data->size();
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