#ifndef MLP_H
#define MLP_H

#include "Module.h"
#include "Linear.h"
#include <vector>

class MLP: public Module {
public:
    MLP(int fan_in, std::vector<int> hidden_sizes, std::string activation_layer="", int seed=42
    // later down the line...
    // bias, dropout, norm_layer, inplace
    );

    std::vector<ValuePtr> forward(const std::vector<ValuePtr>& xin); // forward pass

    std::vector<ValuePtr> parameters() const override; // params

private:
    std::string activation_layer;
    std::vector<Linear> layers {};
};


#endif