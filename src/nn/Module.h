#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include "autograd/Tensor.h"

class Module {
protected:
    // tracks child modules assigned to this module instance
    std::vector<std::shared_ptr<Module>> sub_modules;

public:
    virtual ~Module() = default;

    virtual TensorPtr forward(const TensorPtr& input) = 0;

    // every layer must have this!!
    virtual std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> all_params;
        for (const auto& mod : sub_modules) {
            auto mod_params = mod->parameters();
            all_params.insert(all_params.end(), mod_params.begin(), mod_params.end());
        }
        return all_params;
    }

    // registers a child layer dependency
    void register_submodule(std::shared_ptr<Module> module) {
        sub_modules.push_back(module);
    }

    // function to clear grads before training step
    void zero_grad() {
        for (auto& p : parameters()){
            p->zero_grad();
        }
    }
};

#endif