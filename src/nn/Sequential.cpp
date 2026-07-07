#include "Sequential.h"

void Sequential::add(std::shared_ptr<Module> layer){
    layers.push_back(layer);
}

TensorPtr Sequential::forward(const TensorPtr& input) {
    TensorPtr curr = input;

    for(auto& layer : layers){
        curr = layer->forward(curr);
    }

    return curr;
}

std::vector<TensorPtr> Sequential::parameters() const {
    std::vector<TensorPtr> all_params;
    for (const auto& layer : layers) {
        auto layer_params = layer->parameters();
        all_params.insert(all_params.end(), layer_params.begin(), layer_params.end());
    }
    return all_params;
}