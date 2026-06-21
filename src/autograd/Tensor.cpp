#include "Tensor.h"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// constructor for leaf nodes
Tensor::Tensor(std::vector<double> values, std::vector<size_t> shape, bool requires_grad)
    : data(values), shape(shape), requires_grad(requires_grad), op(""), backward_func([](){}) {
    
    grad.resize(data.size(), 0.0); // resize gradient storage to match data

    // compute row-major strides
    strides.resize(shape.size(), 1);
    if (!shape.empty()){
        for(int i {static_cast<int>(shape.size()) - 2}; i >= 0; --i){
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
}

// graph constructor for operations
Tensor::Tensor(std::vector<double> values, std::vector<size_t> shape, std::set<TensorPtr> children, std::string operation)
    : data(values), shape(shape), requires_grad(false), prev(children), op(operation), backward_func([](){}){

    grad.resize(data.size(), 0.0);

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
        
        out_values[i] = lhs->data[flat_lhs] + rhs->data[flat_rhs];
        
        // advance the coordinate system to the next spatial block
        Tensor::advance_coordinates(current_idx, out_shape);
    }

    // construct graph node
    auto out = std::make_shared<Tensor>(out_values, out_shape, std::set<TensorPtr>{lhs, rhs}, "+");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, out_shape, lhs_b_strides, rhs_b_strides, total_elements]() {
        if(auto out_ptr = weak_out.lock()) {
            std::vector<size_t> back_idx(out_shape.size(), 0);

            for(size_t i {0}; i < total_elements; ++i){
                size_t flat_lhs = Tensor::get_flat_index_from_broadcast(back_idx, lhs_b_strides);
                size_t flat_rhs = Tensor::get_flat_index_from_broadcast(back_idx, rhs_b_strides);
                
                // upstream gradient vector aligns perfectly with total_elements sequence
                auto upstream_grad = out_ptr->grad[i];

                // zero-stride values automatically combine multi-dimensional gradients here
                lhs->grad[flat_lhs] += upstream_grad;
                rhs->grad[flat_rhs] += upstream_grad;

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

        out_values[i] = lhs->data[flat_lhs] - rhs->data[flat_rhs];

        // advance the coordinate system to the next spatial block
        Tensor::advance_coordinates(current_idx, out_shape);
    }

    // construct graph node
    auto out = std::make_shared<Tensor>(out_values, out_shape, std::set<TensorPtr>{lhs,rhs}, "-");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, out_shape, lhs_b_strides, rhs_b_strides, total_elements]() {
        if(auto out_ptr = weak_out.lock()){
            std::vector<size_t> back_idx(out_shape.size(), 0);

            for(size_t i {0}; i < total_elements; ++i){
                size_t flat_lhs = Tensor::get_flat_index_from_broadcast(back_idx, lhs_b_strides);
                size_t flat_rhs = Tensor::get_flat_index_from_broadcast(back_idx, rhs_b_strides);
                
                // upstream gradient vector aligns perfectly with total_elements sequence
                auto upstream_grad = out_ptr->grad[i];

                // zero-stride values automatically combine multi-dimensional gradients here
                lhs->grad[flat_lhs] += upstream_grad;
                rhs->grad[flat_rhs] -= upstream_grad;

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
        
        out_values[i] = lhs->data[flat_lhs] * rhs->data[flat_rhs];
        
        // advance the coordinate system to the next spatial block
        Tensor::advance_coordinates(current_idx, out_shape);
    }

    auto out = std::make_shared<Tensor>(out_values, out_shape, std::set<TensorPtr>{lhs, rhs}, "*");

    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, out_shape, lhs_b_strides, rhs_b_strides, total_elements](){
        if (auto out_ptr = weak_out.lock()){
            std::vector<size_t> back_idx(out_shape.size(), 0);

            for(size_t i {0}; i < total_elements; ++i){
                size_t flat_lhs = Tensor::get_flat_index_from_broadcast(back_idx, lhs_b_strides);
                size_t flat_rhs = Tensor::get_flat_index_from_broadcast(back_idx, rhs_b_strides);
                
                auto upstream_grad = out_ptr->grad[i];

                lhs->grad[flat_lhs] += upstream_grad * rhs->data[flat_rhs];
                rhs->grad[flat_rhs] += upstream_grad * lhs->data[flat_lhs];

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
                    sum_val += lhs->data[lhs_flat] * rhs->data[rhs_flat];
                }
                out_values[batch_out_off + i * N + j] = sum_val;
            }
        }
        Tensor::advance_coordinates(current_batch_idx, batch_shape);
    }

    auto out = std::make_shared<Tensor>(out_values, out_shape, std::set<TensorPtr>{lhs, rhs}, "matmul");

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
                for (size_t i = 0; i < M; ++i) {
                    for (size_t k = 0; k < K; ++k) {
                        double grad_sum = 0.0;
                        for (size_t j = 0; j < N; ++j) {
                            size_t out_flat = batch_out_off + i * N + j;
                            size_t rhs_flat = batch_rhs_off + k * rhs_stride_K + j * rhs_stride_N;
                            grad_sum += out_ptr->grad[out_flat] * rhs->data[rhs_flat];
                        }
                        size_t lhs_flat = batch_lhs_off + i * lhs_stride_M + k * lhs_stride_K;
                        lhs->grad[lhs_flat] += grad_sum; // automatically reduces across broadcasted batch dimensions
                    }
                }

                // dL/dRHS = LHS^T * dL/dOut
                for (size_t k = 0; k < K; ++k) {
                    for (size_t j = 0; j < N; ++j) {
                        double grad_sum = 0.0;
                        for (size_t i = 0; i < M; ++i) {
                            size_t out_flat = batch_out_off + i * N + j;
                            size_t lhs_flat = batch_lhs_off + i * lhs_stride_M + k * lhs_stride_K;
                            grad_sum += lhs->data[lhs_flat] * out_ptr->grad[out_flat];
                        }
                        size_t rhs_flat = batch_rhs_off + k * rhs_stride_K + j * rhs_stride_N;
                        rhs->grad[rhs_flat] += grad_sum; // automatically reduces across broadcasted batch dimensions
                    }
                }

                Tensor::advance_coordinates(back_batch_idx, batch_shape);
            }
        }
    };

    return out;
}

// sum function
TensorPtr Tensor::sum(){
    std::vector<double> out_values(1, 0.0);
    for (double val : this->data) {
        out_values[0] += val;
    }

    auto out = std::make_shared<Tensor>(out_values, std::vector<size_t>{1}, std::set<TensorPtr>{shared_from_this()}, "sum");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();

    out->backward_func = [self, weak_out]() {
        if (auto out_ptr = weak_out.lock()){
            double upstream_grad = out_ptr->grad[0];
            // broadcast the scalar gradient to every single native data location
            for (size_t i = 0; i < self->grad.size(); ++i) {
                self->grad[i] += upstream_grad;
            }
        }
    };

    return out;
}

TensorPtr Tensor::transpose(size_t dim0, size_t dim1) {
    if (dim0 >= shape.size() || dim1 >= shape.size()) {
        throw std::invalid_argument("Transpose dimensions out of bounds");
    }

    std::vector<size_t> new_shape = shape;
    std::vector<size_t> new_strides = strides;

    // swap metadata dimensions
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    // build the new output tensor
    auto out = std::make_shared<Tensor>(data, 
                                        new_shape, 
                                        std::set<TensorPtr>{shared_from_this()}, 
                                        "transpose");
    out->strides = new_strides; // override default contigous layout strides

    // backward pass
    // the derivative of a transpose is simply transposing the upstream grads back
    std::weak_ptr<Tensor> weak_out = out;
    auto self = shared_from_this();
    out->backward_func = [self, weak_out, dim0, dim1](){
        if(auto out_ptr = weak_out.lock()){
            size_t total_elements = out_ptr->data.size();
            std::vector<size_t> back_idx(out_ptr->shape.size(), 0);

            for(size_t i {0}; i < total_elements; ++i){
                // out_flat is perfectly contiguous tracking the odometer step
                size_t out_flat = i;

                // mirror coordinates to look up parent offset
                std::vector<size_t> self_idx = back_idx;
                std::swap(self_idx[dim0], self_idx[dim1]);

                size_t self_flat = self->get_flat_index(self_idx);

                // accum grads back to parent
                self->grad[self_flat] += out_ptr->grad[out_flat];

                // step odometer forward
                Tensor::advance_coordinates(back_idx, out_ptr->shape);
            }
        }
    };

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

    // out node start with grad 1.0
    if (!grad.empty()) {
        std::fill(grad.begin(), grad.end(), 1.0);
    } 

    // process nodes in reverse topo order
    for(auto it = topo.rbegin(); it != topo.rend(); ++it){
        if ((*it)->requires_grad){
            (*it)->backward_func();
        }
    }
}

// add print function later
// void Tensor::print() const {
//     std::cout <<
// }