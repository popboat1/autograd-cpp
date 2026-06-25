#include "Linear.h"
#include <random>

Linear::Linear(int fan_in, int fan_out, int seed) : fan_in(fan_in), fan_out(fan_out){
    // rng for w & b inits
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(-1.0, 0.1);

    // initialize weights matrix
    std::vector<double> w_vals(fan_in * fan_out);
    for (double& val : w_vals) val = dis(gen);
    weights = std::make_shared<Tensor>(w_vals, std::vector<size_t>{(size_t)fan_in, (size_t)fan_out}, true);

    // initialize bias
    std::vector<double> b_vals(fan_out);
    for (double& val : b_vals) val = dis(gen);
    biases = std::make_shared<Tensor>(b_vals, std::vector<size_t>{(size_t)fan_out}, true);
}

TensorPtr Linear::forward(const TensorPtr& xin) {
    // compute scalar dot product for each neuron
    return Tensor::matmul(xin, weights) + biases;
}

std::vector<TensorPtr> Linear::parameters() const {
    return {weights, biases};
}