#ifndef AVGPOOL2D_H
#define AVGPOOL2D_H

#include "nn/Module.h"
#include "autograd/Tensor.h"
#include <vector>

class AvgPool2D : public Module {
public:
    size_t kernel_size;
    size_t stride;

    // constructor setting window sizing limits
    AvgPool2D(size_t kernel_size, size_t stride = 2);

    // tracking forward execution pass
    TensorPtr forward(const TensorPtr& input) override;

    // zero-weight parameter tracker override
    std::vector<TensorPtr> parameters() const override;
};

#endif