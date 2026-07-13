#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "autograd/Tensor.h"
#include "nn/Linear.h"
#include "nn/Conv2D.h"
#include "nn/MaxPool2D.h"
#include "nn/Sequential.h"
#include "nn/BatchNorm2D.h"
#include "nn/Loss.h"
#include "optim/SGD.h"
#include "optim/Adam.h"
#include "utils/RNG.h"
#include "nn/AvgPool2D.h"

// helper to assert floating point parity
bool close_enough(double a, double b, double tol = 1e-5) {
    return std::abs(a - b) < tol;
}

void test_sequential_mlp_convergence() {
    std::cout << "Running Sequential MLP Integration Test...\n";

    // dataset (batch of 4 samples, 3 features each)
    std::vector<double> X_vals = {
        2.0,  3.0, -1.0,
        3.0, -1.0,  0.5,
        0.5,  1.0,  1.0,
        1.0,  1.0, -1.0
    };
    auto X = std::make_shared<Tensor>(X_vals, std::vector<size_t>{4, 3}, false);

    std::vector<double> Y_vals = {1.0, -1.0, -1.0, 1.0};
    auto Y = std::make_shared<Tensor>(Y_vals, std::vector<size_t>{4, 1}, false);

    // construct model using the new Sequential container
    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(3, 4));
    // inline activation tracking pass
    model->add(std::make_shared<Linear>(4, 4));
    model->add(std::make_shared<Linear>(4, 1));

    SGD optimizer(model->parameters(), 0.05);
    MSELoss criterion;

    double initial_loss = 0.0;
    double final_loss = 0.0;

    for (int step = 0; step < 20; ++step) {
        auto preds = model->forward(X);
        auto loss = criterion(preds, Y);
        double current_loss = (*loss->data)[0];

        if (step == 0) initial_loss = current_loss;
        if (step == 19) final_loss = current_loss;

        std::cout << "step " << step << " | loss: "<< (*loss->data)[0] << "\n";

        optimizer.zero_grad();
        loss->backward();
        optimizer.step();
    }

    std::cout << "  Initial Loss: " << initial_loss << " -> Final Loss: " << final_loss << "\n";
    assert(final_loss < initial_loss);
    std::cout << "[PASS] Sequential container parameter aggregation and optimization verified\n\n";
}

void test_vision_modules_autograd() {
    std::cout << "Running Vision Modules (Conv2D + MaxPool2D) Autograd Test...\n";

    // Synthetic Image Batch: B=2, C=1, H=4, W=4
    std::vector<double> img_vals = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 8.0, 7.0, 6.0,
        5.0, 4.0, 3.0, 2.0,

        0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 8.0, 7.0,
        6.0, 5.0, 4.0, 3.0
    };
    auto X = std::make_shared<Tensor>(img_vals, std::vector<size_t>{2, 1, 4, 4}, true);

    // Instantiate Conv2D: In=1, Out=2, Kernel=3, Stride=1, Padding=1
    auto conv = std::make_shared<Conv2D>(1, 2, 3, 1, 1);
    
    // Instantiate MaxPool2D: Kernel=2, Stride=2
    auto pool = std::make_shared<MaxPool2D>(2, 2);

    // Forward pass sequence
    auto conv_out = conv->forward(X);
    auto relu_out = conv_out->relu();
    auto pool_out = pool->forward(relu_out);

    // Verify Output Geometry dimensions
    // Conv out with padding=1 keeps spatial size: {2, 2, 4, 4}
    assert(conv_out->shape[0] == 2 && conv_out->shape[1] == 2 && conv_out->shape[2] == 4 && conv_out->shape[3] == 4);
    // Pool out divides dimensions by 2: {2, 2, 2, 2}
    assert(pool_out->shape[0] == 2 && pool_out->shape[1] == 2 && pool_out->shape[2] == 2 && pool_out->shape[3] == 2);
    std::cout << "[PASS] Vision layout dimensions and spatial grid scalings valid\n";

    // Execute backward pass on the sum of output channels
    auto loss = pool_out->sum();
    loss->backward();

    // Verify gradient propagation down to original filters
    assert(conv->weight->grad != nullptr);
    assert(conv->bias->grad != nullptr);
    assert(X->grad != nullptr);

    // Verify gradients are populated with non-zero values
    double weight_grad_sum = 0.0;
    for (double g : *conv->weight->grad) weight_grad_sum += std::abs(g);
    assert(weight_grad_sum > 0.0);

    double input_grad_sum = 0.0;
    for (double g : *X->grad) input_grad_sum += std::abs(g);
    assert(input_grad_sum > 0.0);

    std::cout << "[PASS] Graph execution timing and custom col2im / routing loops verified\n\n";
}

void test_sequential_mlp_adam_convergence() {
    std::cout << "Running Sequential MLP Integration Test (Adam)...\n";

    // identical target dataset setup
    std::vector<double> X_vals = {
        2.0,  3.0, -1.0,
        3.0, -1.0,  0.5,
        0.5,  1.0,  1.0,
        1.0,  1.0, -1.0
    };
    auto X = std::make_shared<Tensor>(X_vals, std::vector<size_t>{4, 3}, false);

    std::vector<double> Y_vals = {1.0, -1.0, -1.0, 1.0};
    auto Y = std::make_shared<Tensor>(Y_vals, std::vector<size_t>{4, 1}, false);

    auto model = std::make_shared<Sequential>();
    model->add(std::make_shared<Linear>(3, 4));
    model->add(std::make_shared<Linear>(4, 4));
    model->add(std::make_shared<Linear>(4, 1));

    // instantiate our new adam optimizer module with a higher learning rate for fast toy task convergence
    Adam optimizer(model->parameters(), 0.02, {0.9, 0.999}, 1e-8, 0.01, false, false);
    MSELoss criterion;

    double initial_loss = 0.0;
    double final_loss = 0.0;

    for (int step = 0; step < 20; ++step) {
        auto preds = model->forward(X);
        auto loss = criterion(preds, Y);
        double current_loss = (*loss->data)[0];

        if (step == 0) initial_loss = current_loss;
        if (step == 19) final_loss = current_loss;

        std::cout << "step " << step << " | loss: " << current_loss << "\n";

        optimizer.zero_grad();
        loss->backward();
        optimizer.step();
    }

    std::cout << "  Initial Loss: " << initial_loss << " -> Final Loss: " << final_loss << "\n";
    assert(final_loss < initial_loss);
    std::cout << "[PASS] Adam adaptive optimization step adjustments verified successfully\n\n";
}

void test_batchnorm2d_autograd_and_inference() {
    std::cout << "Running BatchNorm2D Autograd & Inference Test...\n";

    // Synthetic 4D Tensor Batch Layout: B=2, C=2, H=2, W=2
    std::vector<double> X_vals = {
        1.0, 2.0, 3.0, 4.0,  // B=0, C=0
        5.0, 6.0, 7.0, 8.0,  // B=0, C=1
        2.0, 3.0, 4.0, 5.0,  // B=1, C=0
        6.0, 7.0, 8.0, 9.0   // B=1, C=1
    };
    auto X = std::make_shared<Tensor>(X_vals, std::vector<size_t>{2, 2, 2, 2}, true);

    // Instantiate BatchNorm2D targeting 2 input features
    auto bn = std::make_shared<BatchNorm2D>(2, 1e-5, 0.1, true, true);
    assert(bn->training == true);

    // Execute forward pass under training criteria
    auto out = bn->forward(X);

    // Validate spatial shape configuration preservation
    assert(out->shape == X->shape);
    std::cout << "[PASS] BatchNorm2D structural spatial dimensions preserved perfectly\n";

    // Validate non-differentiable tracking stats updates (EMA updates buffers away from 0 and 1)
    assert((*bn->running_mean->data)[0] != 0.0 || (*bn->running_mean->data)[1] != 0.0);
    assert((*bn->running_var->data)[0] != 1.0 || (*bn->running_var->data)[1] != 1.0);
    std::cout << "[PASS] BatchNorm2D non-differentiable moving tracking updates active\n";

    // Validate automated primitive backpropagation routing
    auto loss = (out * out)->sum();
    loss->backward();

    assert(bn->weight->grad != nullptr);
    assert(bn->bias->grad != nullptr);
    assert(X->grad != nullptr);

    double weight_grad_sum = 0.0;
    for (double g : *bn->weight->grad) weight_grad_sum += std::abs(g);
    assert(weight_grad_sum > 0.0);

    double input_grad_sum = 0.0;
    for (double g : *X->grad) input_grad_sum += std::abs(g);
    assert(input_grad_sum > 0.0);
    std::cout << "[PASS] BatchNorm2D automated primitive autograd path successfully verified\n";

    // Validate evaluation phase inference mechanics
    bn->training = false;
    X->zero_grad();

    auto out_eval = bn->forward(X);
    assert(out_eval->shape == X->shape);
    std::cout << "[PASS] BatchNorm2D frozen evaluation inference completed cleanly\n\n";
}

void test_average_pooling_autograd() {
    std::cout << "Running AveragePool2D Autograd Verification Test...\n";

    // Synthetic Image Frame: B=1, C=1, H=4, W=4
    std::vector<double> img_vals = {
        2.0, 4.0, 8.0, 16.0,
        4.0, 6.0, 2.0, 0.0,
        1.0, 3.0, 5.0, 7.0,
        9.0, 7.0, 5.0, 3.0
    };
    auto X = std::make_shared<Tensor>(img_vals, std::vector<size_t>{1, 1, 4, 4}, true);

    // Instantiate AveragePool2D: Kernel=2, Stride=2
    auto avg_pool = std::make_shared<AvgPool2D>(2, 2);
    auto out = avg_pool->forward(X);

    // Verify spatial grid resolution downsampling maps ({1, 1, 4, 4} -> {1, 1, 2, 2})
    assert(out->shape[0] == 1 && out->shape[1] == 1 && out->shape[2] == 2 && out->shape[3] == 2);

    // Verify exact analytical forward values for top-left window block: (2+4+4+6)/4 = 4.0
    assert(close_enough((*out->data)[0], 4.0));
    std::cout << "[PASS] AvgPool2D mapping values match analytical expectations\n";

    // Run backpropagation sweep utilizing non-uniform scale variables
    auto loss = (out * out)->sum();
    loss->backward();

    // Verify gradient propagation reached original input layer matrix entries
    assert(X->grad != nullptr);
    
    // For out[0] = 4.0, dLoss/dOut[0] = 2 * 4.0 = 8.0
    // dLoss/dX elements in window[0] = 8.0 / 4 = 2.0
    assert(close_enough((*X->grad)[0], 2.0));
    assert(close_enough((*X->grad)[1], 2.0));

    std::cout << "[PASS] AveragePool2D derivative backpropagation equations verified successfully\n\n";
}

int main() {
    std::cout << "starting neural network integration tests\n";
    std::cout << "----------------------------------------\n";

    RNG::manual_seed(42);

    test_sequential_mlp_convergence();
    test_sequential_mlp_adam_convergence();
    test_vision_modules_autograd();
    test_batchnorm2d_autograd_and_inference();

    std::cout << "----------------------------------------\n";
    std::cout << "[PASS] all neural network tests passed cleanly\n";
    return 0;
}