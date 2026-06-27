#include "SGD.h"
#include <vector>

SGD::SGD(std::vector<TensorPtr> params, double lr, double momentum, double weight_decay)
    : params(params), lr(lr), momentum_factor(momentum), wd(weight_decay) {
    
    // allocate velocity tracking arrays identical to the flat size of each parameter
    for (const auto& p : params) {
        velocities.push_back(std::make_shared<Tensor>(std::vector<double>(p->data->size(), 0.0), p->shape, false));
    }
}

void SGD::step(){
    for(size_t i {0}; i < params.size(); ++i){
        if (params[i]->grad == nullptr) continue;

        // create a zero-copy tensor wrapper around the parameter's underlying gradient vector array
        auto grad_tensor = std::make_shared<Tensor>(params[i]->grad, nullptr, params[i]->shape, std::vector<TensorPtr>{}, "grad_wrapper");
        
        if (wd > 0.0) {
            grad_tensor->add_(params[i] * wd);
        }

        // update velocity V = (V * momentum) + Grad
        velocities[i] = (velocities[i] * momentum_factor) + grad_tensor;

        // W -= V * lr
        params[i]->sub_(velocities[i] * lr);
    }
}

void SGD::zero_grad() {
    for(auto& p : params){
        p->zero_grad();
    }
}