#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <set>

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    // properties
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    // autograd properties
    bool requires_grad;
    std::string op;
    std::set<TensorPtr> prev;
    std::function<void()> backward_func;

    // constructors
    Tensor(std::vector<double> values, std::vector<size_t> shape, bool requires_grad = true);
    Tensor(std::vector<double> values, std::vector<size_t> shape, std::set<TensorPtr> children, std::string operation = "");

    // helper to map multidimensional index arrays to 1D flat storage offset
    size_t get_flat_index(const std::vector<size_t>& indices) const;

    void backward();

    // operations
    static TensorPtr matmul(const TensorPtr& lhs, const TensorPtr& rhs);
    TensorPtr transpose();
    TensorPtr sum();
};

#endif