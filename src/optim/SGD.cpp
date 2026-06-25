#include "SGD.h"
#include <vector>

SGD::SGD(std::vector<TensorPtr> params, double lr, double momentum, double weight_decay)
    : params(params), lr(lr), momentum_factor(momentum), wd(weight_decay) {
    
    // allocate velocity tracking arrays identical to the flat size of each parameter
    for (const auto& p : params) {
        velocities.push_back(std::vector<double>(p->data->size(), 0.0));
    }
}

void SGD::step(){
    for(size_t i {0}; i < params.size(); ++i){
        for(size_t j {0}; j < params[i]->data->size(); ++j){
            double gradient = (*params[i]->grad)[j];

            if(wd > 0.0){
                gradient += (wd * (*params[i]->data)[j]);
            }

            velocities[i][j] = (momentum_factor * velocities[i][j]) + gradient;

            (*params[i]->data)[j] -= (lr * velocities[i][j]);
        }
    }
}

void SGD::zero_grad() {
    for(auto& p : params){
        std::fill(p->grad->begin(), p->grad->end(), 0.0);
    }
}