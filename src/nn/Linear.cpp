#include "Linear.h"
#include <random>

Linear::Linear(int fan_in, int fan_out, int seed) : fan_in(fan_in), fan_out(fan_out){
    // rng for w & b inits
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(-1.0, 0.1);

    // initialize weights matrix
    weights.resize(fan_out);
    for(int i {0}; i < fan_out; ++i){
        weights[i].resize(fan_in);
        for(int j {0}; j < fan_in; ++j){
            weights[i][j] = make_val(dis(gen));
        }
    }

    // initialize bias
    biases.resize(fan_out);
    for(int i {0}; i < fan_out; ++i){
        biases[i] = make_val(dis(gen));
    }
}

std::vector<ValuePtr> Linear::forward(const std::vector<ValuePtr>& xin){
    std::vector<ValuePtr> out(fan_out);

    // compute scalar dot product for each neuron
    for (int i {0}; i < fan_out; ++i){
        ValuePtr sum = make_val(0.0);
        for(int j {0}; j < fan_in; ++j){
            sum = sum + (weights[i][j] * xin[j]);
        }
        out[i] = sum + biases[i];
    }

    return out;
}

std::vector<ValuePtr> Linear::parameters() const {
    std::vector<ValuePtr> params;
    
    for (const auto& row : weights) {
        for (const auto& w : row) {
            params.push_back(w);
        }
    }
    for (const auto& b : biases) {
        params.push_back(b);
    }
    
    return params;
}