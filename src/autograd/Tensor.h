#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <functional>
#include <string>

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

    
    // store calculated variables for dimensional reductions
    struct ReductionMeta {
        std::vector<size_t> out_shape;
        size_t total_out_elements;
        size_t reduced_size;
        size_t inner_block_size;
        size_t outer_block_size;
    };
    
    // helper function to calculate all the boilerplate metadata
    ReductionMeta prepare_reduction_metadata(size_t dim, bool keepdim) const;
    
    
    
public:
    // advances an N-dimensional coordinate vector right-to-left; returns false when fully complete
    static bool advance_coordinates(
        std::vector<size_t>& coords, 
        const std::vector<size_t>& target_shape
    );
    
    // properties
    std::shared_ptr<std::vector<double>> data;
    std::shared_ptr<std::vector<double>> grad;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    // autograd properties
    bool requires_grad;
    std::string op;
    std::vector<TensorPtr> prev;
    std::function<void()> backward_func;

    // constructors
    Tensor(std::vector<double> values, std::vector<size_t> shape, bool requires_grad = true);
    Tensor(std::vector<double> values, 
           std::vector<size_t> shape, 
           std::vector<TensorPtr> children, 
           std::string operation = "");
    
    // internal zero-copy constructor
    Tensor(std::shared_ptr<std::vector<double>> shared_data, 
           std::shared_ptr<std::vector<double>> shared_grad, 
           std::vector<size_t> shape, 
           std::vector<TensorPtr> children, 
           std::string operation = "");

    // helper to map multidimensional index arrays to 1D flat storage offset
    size_t get_flat_index(const std::vector<size_t>& indices) const;
    bool is_contiguous() const;
    TensorPtr contiguous();
    void backward();

    // in-place mutators & encapsulated ops
    TensorPtr add_(const TensorPtr& other);
    TensorPtr sub_(const TensorPtr& other);
    void zero_grad();

    // operations
    static TensorPtr matmul(const TensorPtr& lhs, const TensorPtr& rhs);
    friend TensorPtr operator+(const TensorPtr& lhs, const TensorPtr& rhs);
    friend TensorPtr operator-(const TensorPtr& lhs, const TensorPtr& rhs);
    friend TensorPtr operator*(const TensorPtr& lhs, const TensorPtr& rhs);
    friend TensorPtr operator/(const TensorPtr& lhs, const TensorPtr& rhs);
    TensorPtr transpose(size_t dim0, size_t dim1);
    TensorPtr sum();

    // tensor comparison operators
    friend TensorPtr operator==(const Tensor& lhs, const Tensor& rhs);
    friend TensorPtr operator<(const Tensor& lhs, const Tensor& rhs);
    friend TensorPtr operator>(const Tensor& lhs, const Tensor& rhs);

    // unary ops
    TensorPtr sqrt();
    TensorPtr neg();
    friend TensorPtr operator-(const TensorPtr& tensor); // prefix negation

    // tensor * scalar multiplications
    friend TensorPtr operator*(const TensorPtr& lhs, double rhs);
    friend TensorPtr operator*(double lhs, const TensorPtr& rhs);
    
    // more element wise math ops & activation fns
    TensorPtr relu();
    TensorPtr exp();
    TensorPtr tanh();
    TensorPtr sigmoid();
    TensorPtr log();
    TensorPtr pow(double exponent);

    // advanced activations
    TensorPtr softmax(size_t dim);
    TensorPtr log_softmax(size_t dim);
    
    // dimensional reduction ops
    TensorPtr sum(size_t dim, bool keepdim = false);
    TensorPtr mean(size_t dim, bool keepdim = false);
    TensorPtr max(size_t dim, bool keepdim = false);
    TensorPtr min(size_t dim, bool keepdim = false);
    TensorPtr argmax(size_t dim, bool keepdim = false);
    TensorPtr argmin(size_t dim, bool keepdim = false);

    // tensor shape manipulation ops
    TensorPtr reshape(const std::vector<size_t>& new_shape);
    TensorPtr squeeze(size_t dim);
    TensorPtr unsqueeze(size_t dim);
    TensorPtr permute(const std::vector<size_t>& dims);

    // reshapes tensor view without memory copies; accepts a single -1 placeholder axis
    TensorPtr view(const std::vector<int>& target_shape);

    // tensor broadcasting expansion view
    TensorPtr expand(const std::vector<size_t>& new_shape);

    // tensor sorting utilities
    TensorPtr argsort(size_t dim, bool descending = false);

    // prints metadata footprint and formatted multi-dimensional bracket layouts
    void print() const;

    void ensure_grad_allocated();
};

#endif