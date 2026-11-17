#ifndef MODULE_H
#define MODULE_H

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include "autograd/Tensor.h"

class Module {
protected:
    // tracks child modules assigned to this module instance
    std::vector<std::shared_ptr<Module>> sub_modules;

public:
    virtual ~Module() = default;

    bool training = true;
    virtual void train(bool mode = true)
    {
        this->training = mode;

        for(auto& mod : sub_modules)
        {
            if(mod) mod->train(mode);
        }
    }

    void eval() { train(false); }

    virtual TensorPtr forward(const TensorPtr& input) = 0;

    // every layer must have this!!
    virtual std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> all_params;
        for (const auto& mod : sub_modules) {
            if (!mod) continue;
            auto mod_params = mod->parameters();
            all_params.insert(all_params.end(), mod_params.begin(), mod_params.end());
        }
        return all_params;
    }

    // registers a child layer dependency without duplicating entries
    void register_submodule(std::shared_ptr<Module> module) {
        if (!module) return;
        if (std::find(sub_modules.begin(), sub_modules.end(), module) == sub_modules.end()) {
            sub_modules.push_back(std::move(module));
        }
    }

    // function to clear grads before training step
    void zero_grad() {
        for (auto& p : parameters()){
            p->zero_grad();
        }
    }

    virtual void to(Device target_device) {
        for (auto& p : parameters()) {
            p->to(target_device);
        }

        for(auto& mod : sub_modules)
        {
            if(mod)
            {
                mod->to(target_device);
            }
        }
    }
};

#endif