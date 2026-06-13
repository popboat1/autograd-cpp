#include "MLP.h"
#include <random>
#include <vector>

MLP::MLP(int fan_in, std::vector<int> hidden_sizes, std::string activation_layer, int seed){
    this->activation_layer = activation_layer;
    int current_in = fan_in;
    int seed_modifier = {0}; // to break symmetry between layers

    for(int& hidden_size : hidden_sizes){
        layers.emplace_back(current_in, hidden_size, seed + seed_modifier);
        current_in=hidden_size;
        ++seed_modifier;
    }
}

std::vector<ValuePtr> MLP::forward(const std::vector<ValuePtr>& xin){
    std::vector<ValuePtr> current_signals = xin;

    for(size_t i {0}; i < layers.size(); ++i){
        auto out = layers[i].forward(current_signals);

        if(i < layers.size() - 1){
            for(auto& val : out){
                if(activation_layer == "relu"){
                    val = val->relu();
                }else if(activation_layer == "tanh"){
                    val = val->tanh();
                }
            }
        }
        current_signals = out;
    }
    return current_signals;
}

std::vector<ValuePtr> MLP::parameters() const {
    std::vector<ValuePtr> total_params;

    for(const auto& layer : layers){
        auto layer_params = layer.parameters();
        total_params.insert(total_params.end(), layer_params.begin(), layer_params.end());
    }

    return total_params;
}