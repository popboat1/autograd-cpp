#include "MLP.h"
#include <random>
#include <vector>

MLP::MLP(int fan_in, std::vector<int> hidden_sizes, std::string activation_layer){
    this->activation_layer = activation_layer;
    int current_in = fan_in;
    int seed_modifier = 0; // to break symmetry between layers

    for(int hidden_size : hidden_sizes) {
        layers.emplace_back(current_in, hidden_size, "kaiming");
        current_in = hidden_size;
    }
}

TensorPtr MLP::forward(const TensorPtr& xin) {
    TensorPtr current_signal = xin;

    for(size_t i {0}; i < layers.size(); ++i){
        current_signal = layers[i].forward(current_signal);

        if(i < layers.size() - 1){
            if(activation_layer == "relu"){
                current_signal = current_signal->relu();
            } else if(activation_layer == "tanh"){
                current_signal = current_signal->tanh();
            }
        }
    }
    return current_signal;
}

std::vector<TensorPtr> MLP::parameters() const {
    std::vector<TensorPtr> total_params;
    for(const auto& layer : layers){
        auto layer_params = layer.parameters();
        total_params.insert(total_params.end(), layer_params.begin(), layer_params.end());
    }
    return total_params;
}