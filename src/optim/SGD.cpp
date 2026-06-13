#include "SGD.h"
#include <vector>

SGD::SGD(std::vector<ValuePtr> params, double lr, double momentum, double weight_decay)
    : params(params), lr(lr), momentum_factor(momentum), wd(weight_decay), velocities(params.size(), 0.0){
}

void SGD::step(){
    for(size_t i {0}; i < params.size(); ++i){
        double gradient = params[i]->grad;

        if(wd > 0.0){
            gradient = gradient + (wd * params[i]->data);
        }

        velocities[i] = (momentum_factor * velocities[i]) + gradient;

        gradient = velocities[i];

        params[i]->data = params[i]->data - (lr * gradient);
    }
}

void SGD::zero_grad(){
    for(auto& p : params){
        p->grad = 0.0;
    }
}