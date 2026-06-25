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

    TensorPtr forward(const TensorPtr& xin); // forward pass
    std::vector<TensorPtr> parameters() const override; // params

private:
    std::string activation_layer;
    std::vector<Linear> layers {};
};


#endif