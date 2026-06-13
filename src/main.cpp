#include <iostream>
#include <vector>
#include "autograd/Value.h"
#include "nn/MLP.h"
#include "optim/SGD.h"

int main() {
    // mock input features (3 items) and a target scalar label
    std::vector<ValuePtr> inputs = { make_val(1.0), make_val(-2.0), make_val(0.5) };
    auto target = make_val(1.0); //  we want out net to learn output of 1.0

    // MLP: 3 inputs -> hidden sizes [4, 2] -> 1 final output
    // Uses "relu" hidden activations and an empty string "" for raw output pass-through
    MLP model(3, {4, 2, 1}, "relu", 42);

    // SGD optimizer with learning_rate=0.05, momentum=0.9, weight_decay=1e-4
    SGD optimizer(model.parameters(), 0.05, 0.9, 1e-4);

    auto sample_param = model.parameters()[0];
    std::cout << "total model parameters registered: " << model.parameters().size() << "\n";

    for(int epoch {1}; epoch <= 3; ++epoch){
        optimizer.zero_grad();

        auto outs = model.forward(inputs);
        auto pred = outs[0];

        auto diff = pred - target;
        auto loss = diff->pow(2);

        loss->backward();

        optimizer.step();

        std::cout << "epoch: " << epoch << "| loss: " << loss->data << "| model pred: " << pred->data << "| sample grad: " << sample_param->grad << '\n';
    }
    
    return 0;
}
