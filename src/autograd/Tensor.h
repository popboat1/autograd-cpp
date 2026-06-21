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
private:
    // computes broadcasted shapes and sets strides to 0 along stretched dimensions
    static void compute_broadcast_metadata(
        const TensorPtr& lhs, const TensorPtr& rhs,
        std::vector<size_t>& out_shape,
        std::vector<size_t>& lhs_b_strides,
        std::vector<size_t>& rhs_b_strides
    );

    // maps and N-dimensional odometer coordinate directly to a flat memory address using broadcast strides
    static size_t get_flat_index_from_broadcast(
        const std::vector<size_t>& current_coords,
        const std::vector<size_t>& broadcast_strides
    );

    // advances an N-dimensional coordinate vector right-to-left; returns false when fully complete
    static bool advance_coordinates(
        std::vector<size_t>& coords, 
        const std::vector<size_t>& target_shape
    );

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
    friend TensorPtr operator+(const TensorPtr& lhs, const TensorPtr& rhs);
    friend TensorPtr operator-(const TensorPtr& lhs, const TensorPtr& rhs);
    friend TensorPtr operator*(const TensorPtr& lhs, const TensorPtr& rhs);
    TensorPtr transpose(size_t dim0, size_t dim1);
    TensorPtr sum();

    // reshapes tensor view without memory copies; accepts a single -1 placeholder axis
    TensorPtr view(const std::vector<int>& target_shape);

    // prints metadata footprint and formatted multi-dimensional bracket layouts
    void print() const;
};

#endif