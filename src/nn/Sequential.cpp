#include "Sequential.h"
#include <utility>

void Sequential::add(std::shared_ptr<Module> layer){
    if (!layer) return;
    layers.push_back(layer);
    register_submodule(std::move(layer));
}

TensorPtr Sequential::forward(const TensorPtr& input) {
    TensorPtr curr = input;

    for(auto& layer : layers){
        if (layer) {
            curr = layer->forward(curr);
        }
    }

    return curr;
}
