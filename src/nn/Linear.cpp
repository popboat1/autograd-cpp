#include "Linear.h"
#include "utils/RNG.h"
#include <random>
#include <cmath>
#include <stdexcept>

Linear::Linear(int fan_in, int fan_out, const std::string& init_type) 
    : fan_in(fan_in), fan_out(fan_out) {
    // query the centralized global engine reference
    auto& gen = RNG::get_engine();
    double std_dev = 0.0;

    if (init_type == "kaiming") {
        std_dev = std::sqrt(2.0 / static_cast<double>(fan_in));
    } else if (init_type == "xavier") {
        std_dev = std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
    } else {
        throw std::invalid_argument("unknown initialization type: " + init_type);
    }

    std::normal_distribution<double> dist(0.0, std_dev);

    // initialize weights matrix
    std::vector<double> w_vals(fan_in * fan_out);
    for (double& val : w_vals) val = dist(gen);
    weights = std::make_shared<Tensor>(w_vals, std::vector<size_t>{(size_t)fan_in, (size_t)fan_out}, true);

    // initialize bias
    std::vector<double> b_vals(fan_out);
    biases = std::make_shared<Tensor>(b_vals, std::vector<size_t>{(size_t)fan_out}, true);
}

TensorPtr Linear::forward(const TensorPtr& xin) {
    // compute scalar dot product for each neuron
    return Tensor::matmul(xin, weights) + biases;
}

std::vector<TensorPtr> Linear::parameters() const {
    return {weights, biases};
}