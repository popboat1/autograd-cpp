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

// addition op
TensorPtr operator+(const TensorPtr& lhs, const TensorPtr& rhs){
    if(lhs->shape.size() != 2 || rhs->shape.size() != 2){
        throw std::invalid_argument("both input must be 2d matrices");
    }
    if(lhs->shape != rhs->shape){
        throw std::invalid_argument("shape must match!");
    }

    // alloc a flat memory for out data
    size_t rows = lhs->shape[0];
    size_t cols = lhs->shape[1];
    std::vector<double> out_values(rows*cols, 0.0);

    for(size_t i {0}; i < rows; ++i){
        for(size_t j {0}; j < cols; ++j){
            size_t flat_lhs = lhs->get_flat_index({i, j});
            size_t flat_rhs = rhs->get_flat_index({i, j});
            out_values[i * cols + j] = lhs->data[flat_lhs] + rhs->data[flat_rhs];
        }
    }

    auto out = std::make_shared<Tensor>(out_values, lhs->shape, std::set<TensorPtr>{lhs, rhs}, "+");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, rows, cols]() {
        if(auto out_ptr = weak_out.lock()) {
            for(size_t i {0}; i < rows; ++i){
                for(size_t j {0}; j < cols; ++j){
                    size_t flat_lhs = lhs->get_flat_index({i, j});
                    size_t flat_rhs = rhs->get_flat_index({i, j});
                    auto upstream_grad = out_ptr->grad[i * cols + j];

                    lhs->grad[flat_lhs] += upstream_grad;
                    rhs->grad[flat_rhs] += upstream_grad;
                }
            }
        }
    };

    return out;
}

// substraction op
TensorPtr operator-(const TensorPtr& lhs, const TensorPtr& rhs){
    if(lhs->shape.size() != 2 || rhs->shape.size() != 2){
        throw std::invalid_argument("both input must be 2d matrices");
    }

    if(lhs->shape != rhs->shape){
        throw std::invalid_argument("shape must match!");
    }

    // alloc flat memory for out data
    size_t rows = lhs->shape[0];
    size_t cols = lhs->shape[1];
    std::vector<double> out_values(rows*cols, 0.0);

    for(size_t i {0}; i < rows; ++i){
        for(size_t j {0}; j < cols; ++j){
            size_t flat_lhs = lhs->get_flat_index({i, j});
            size_t flat_rhs = rhs->get_flat_index({i,j});
            out_values[i * cols + j] = lhs->data[flat_lhs] - rhs->data[flat_rhs];
        }
    }

    auto out = std::make_shared<Tensor>(out_values, lhs->shape, std::set<TensorPtr>{lhs,rhs}, "-");

    // backward pass
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, rows, cols]() {
        if(auto out_ptr = weak_out.lock()){
            for(size_t i {0}; i < rows; ++i){
                for(size_t j {0}; j < cols; ++j){
                    size_t flat_lhs = lhs->get_flat_index({i, j});
                    size_t flat_rhs = rhs->get_flat_index({i, j});
                    auto upstream_grad = out_ptr->grad[i * cols + j];

                    lhs->grad[flat_lhs] += upstream_grad;
                    rhs->grad[flat_rhs] -= upstream_grad;
                }
            }
        }
    };

    return out;
}

// matmul op
// TODO: implement batched matmul to support N-dimensional tensors
TensorPtr Tensor::matmul(const TensorPtr& lhs, const TensorPtr& rhs){
    // verify dimensions
    if (lhs->shape.size() != 2 || rhs->shape.size() != 2){
        throw std::invalid_argument("both inputs must be 2d matrices.");
    }
    if(lhs->shape[1] != rhs->shape[0]){
        throw std::invalid_argument("inner dimensions must match.");
    }

    size_t M = lhs->shape[0];
    size_t K = lhs->shape[1];
    size_t N = rhs->shape[1];

    // alloc a flat memory for out matrix data
    std::vector<double> out_values(M * N, 0.0);
    std::vector<size_t> out_shape = {M, N};

    // calculate standard row-column dot products
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < K; ++k) {
                size_t lhs_flat = lhs->get_flat_index({i, k});
                size_t rhs_flat = rhs->get_flat_index({k, j});
                sum += lhs->data[lhs_flat] * rhs->data[rhs_flat];
            }
            out_values[i * N + j] = sum;
        }
    }

    // construct the intermediate graph tracking node
    auto out = std::make_shared<Tensor>(out_values, out_shape, std::set<TensorPtr>{lhs, rhs}, "matmul");

    // capture shared pointers and dimension footprints to isolate graph history lines
    std::weak_ptr<Tensor> weak_out = out;
    out->backward_func = [lhs, rhs, weak_out, M, K, N]() {
        if (auto out_ptr = weak_out.lock()) {
            
            // propagate gradients using transposed matrix calculus combinations
            // dL/dLHS = dL/dOut * RHS^T
            for (size_t i = 0; i < M; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    double grad_sum = 0.0;
                    for (size_t j = 0; j < N; ++j) {
                        size_t out_flat = i * N + j;
                        size_t rhs_flat = rhs->get_flat_index({k, j});
                        grad_sum += out_ptr->grad[out_flat] * rhs->data[rhs_flat];
                    }
                    lhs->grad[lhs->get_flat_index({i, k})] += grad_sum;
                }
            }

            // dL/dRHS = LHS^T * dL/dOut
            for (size_t k = 0; k < K; ++k) {
                for (size_t j = 0; j < N; ++j) {
                    double grad_sum = 0.0;
                    for (size_t i = 0; i < M; ++i) {
                        size_t out_flat = i * N + j;
                        size_t lhs_flat = lhs->get_flat_index({i, k});
                        grad_sum += lhs->data[lhs_flat] * out_ptr->grad[out_flat];
                    }
                    rhs->grad[rhs->get_flat_index({k, j})] += grad_sum;
                }
            }
        }
    };

    return out;
}

// TODO: implement backward func, transpose, sum, addition, substrac, and etc...
TensorPtr Tensor::transpose() {
    if(shape.size() != 2){
        throw std::invalid_argument("transpose error");
    }

    // create swapped shape and size configs
    std::vector<size_t> new_shape = {shape[1], shape[0]};
    std::vector<size_t> new_strides = {strides[1], strides[0]};

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
    out->backward_func = [self, weak_out](){
        if(auto out_ptr = weak_out.lock()){
            //map the output tensor gradients back to self using swapped coordinate tracking
            for(size_t i {0}; i < out_ptr->shape[0]; ++i){
                for(size_t j {0}; j < out_ptr->shape[1]; ++j){
                    size_t out_flat = out_ptr->get_flat_index({i, j});
                    size_t self_flat = self->get_flat_index({j, i});
                    self->grad[self_flat] += out_ptr->grad[out_flat];
                }
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