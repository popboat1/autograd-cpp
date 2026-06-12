#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include "autograd/Value.h"

class Module {
public:
    virtual ~Module() = default;

    // every layer must have this!!
    virtual std::vector<ValuePtr> parameters() const = 0;

    // function to clear grads before training step
    void zero_grad() {
        for (auto& p : parameters()){
            p->grad = 0.0;
        }
    }
};

#endif