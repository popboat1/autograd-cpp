#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include "autograd/Tensor.h"

class Module {
public:
    virtual ~Module() = default;

    // every layer must have this!!
    virtual std::vector<TensorPtr> parameters() const = 0;

    // function to clear grads before training step
    void zero_grad() {
        for (auto& p : parameters()){
            std::fill(p->grad->begin(), p->grad->end(), 0.0);
        }
    }
};

#endif