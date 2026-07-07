#ifndef CONV2D_H
#define CONV2D_H

#include "nn/Module.h"
#include "autograd/Tensor.h"
#include <vector>

class Conv2D : public Module {
private:
    std::vector<double> im2col(
        const TensorPtr& input, 
        size_t out_h, 
        size_t out_w
    ) const;

    void col2im(
        const std::vector<double>& col_grad, 
        const TensorPtr& input_grad, 
        size_t out_h, 
        size_t out_w
    ) const;

public:
    TensorPtr weight;
    TensorPtr bias;
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;

    // constructor initializing dimensions and kaiming he weights
    Conv2D(
        size_t in_channels, 
        size_t out_channels, 
        size_t kernel_size, 
        size_t stride = 1, 
        size_t padding = 0
    );

    // tracking forward execution pass
    TensorPtr forward(const TensorPtr& input) override;

    // module parameter collector override
    std::vector<TensorPtr> parameters() const override;
};

#endif