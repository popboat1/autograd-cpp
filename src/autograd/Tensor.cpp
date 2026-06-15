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

// matmul op
// TODO: implement batched matmul to support N-dimensional tensors
static TensorPtr matmul(const TensorPtr& lhs, const TensorPtr& rhs){
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