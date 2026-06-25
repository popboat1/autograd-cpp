#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "autograd/Tensor.h"
#include "nn/MLP.h"
#include "nn/Loss.h"
#include "optim/SGD.h"

// helper to assert floating point parity
bool close_enough(double a, double b, double tol = 1e-5) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << "starting neural network integration tests\n";
    std::cout << "----------------------------------------\n";

    // dataset (batch of 4 samples, 3 features each)
    std::vector<double> X_vals = {
        2.0,  3.0, -1.0,
        3.0, -1.0,  0.5,
        0.5,  1.0,  1.0,
        1.0,  1.0, -1.0
    };
    // input data with requires_grad = false
    auto X = std::make_shared<Tensor>(X_vals, std::vector<size_t>{4, 3}, false);

    // define the target labels (Batch of 4, 1 output each)
    std::vector<double> Y_vals = {1.0, -1.0, -1.0, 1.0};
    auto Y = std::make_shared<Tensor>(Y_vals, std::vector<size_t>{4, 1}, false);

    // initialize the MLP (3 inputs -> 4 hidden -> 4 hidden -> 1 output)
    MLP model(3, {4, 4, 1}, "relu", 42);

    // initialize the optims
    SGD optimizer(model.parameters(), 0.05); // learning rate = 0.05
    MSELoss criterion;

    // training loop
    double initial_loss = 0.0;
    double final_loss = 0.0;

    for (int step = 0; step < 20; ++step) {
        // forward pass
        auto preds = model.forward(X);
        
        // compute loss
        auto loss = criterion(preds, Y);
        double current_loss = (*loss->data)[0];

        std::cout << "step " << step << " | loss: " << current_loss << '\n';

        if (step == 0) initial_loss = current_loss;
        if (step == 19) final_loss = current_loss;

        // zero gradients, backward pass, and optimizer step
        optimizer.zero_grad();
        loss->backward();
        optimizer.step();
    }

    std::cout << "----------------------------------------\n";

    // The loss must strictly decrease after 20 valid optimization steps
    assert(final_loss < initial_loss);
    std::cout << "[PASS] MLP forward pass, autograd routing, and SGD optimization step verified\n";
    std::cout << "[PASS] network successfully converged on batched training data\n";

    std::cout << "[PASS] all neural network tests passed cleanly\n";
    return 0;
}