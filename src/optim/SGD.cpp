#include "SGD.h"
#include "autograd/Tensor.h"
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

        // handle rare non-contiguous parameters using coordinate maps
        if (!params[i]->is_contiguous()) {
            std::vector<size_t> coords(params[i]->shape.size(), 0);
            size_t total_elements = params[i]->data->size();
            
            for (size_t j = 0; j < total_elements; ++j) {
                size_t flat_idx = params[i]->get_flat_index(coords);
                
                double grad = (*params[i]->grad)[flat_idx];
                if (wd > 0.0) {
                    grad += wd * (*params[i]->data)[flat_idx];
                }
                
                // velocities[i] is guaranteed contiguous by the constructor layout allocation
                (*velocities[i]->data)[j] = (momentum_factor * (*velocities[i]->data)[j]) + grad;
                (*params[i]->data)[flat_idx] -= lr * (*velocities[i]->data)[j];
                
                Tensor::advance_coordinates(coords, params[i]->shape);
            }
        } else {
            size_t total_elements = params[i]->data->size();
            for (size_t j = 0; j < total_elements; ++j) {
                double grad = (*params[i]->grad)[j];
                if (wd > 0.0) {
                    grad += (wd * (*params[i]->data)[j]);
                }
                
                (*velocities[i]->data)[j] = (momentum_factor * (*velocities[i]->data)[j]) + grad;
                (*params[i]->data)[j] -= (lr * (*velocities[i]->data)[j]);
            }
        }
    }
}

void SGD::zero_grad() {
    for(auto& p : params){
        p->zero_grad();
    }
}