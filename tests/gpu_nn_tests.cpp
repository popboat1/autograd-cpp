#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>

#include "autograd/Tensor.h"
#include "nn/Module.h"
#include "nn/Conv2D.h"
#include "nn/BatchNorm2D.h"
#include "nn/MaxPool2D.h"
#include "nn/Linear.h"
#include "nn/Loss.h"
#include "optim/Adam.h"
#include "optim/SGD.h"
#include "utils/cuda_utils.h"

// custom convolutional classification model
class ConvClassifier : public Module {
public:
    std::shared_ptr<Conv2D> conv1;
    std::shared_ptr<BatchNorm2D> bn1;
    std::shared_ptr<MaxPool2D> pool;
    std::shared_ptr<Linear> fc;

    ConvClassifier(size_t in_c, size_t out_c, size_t num_classes) {
        // [B, 1, 8, 8] -> [B, 4, 8, 8]
        conv1 = std::make_shared<Conv2D>(in_c, out_c, 3, 1, 1);
        bn1 = std::make_shared<BatchNorm2D>(out_c);
        // [B, 4, 8, 8] -> [B, 4, 4, 4]
        pool = std::make_shared<MaxPool2D>(2, 2);
        // flattened: 4 * 4 * 4 = 64 features -> num_classes
        fc = std::make_shared<Linear>(out_c * 4 * 4, num_classes);

        register_submodule(conv1);
        register_submodule(bn1);
        register_submodule(pool);
        register_submodule(fc);
    }

    TensorPtr forward(const TensorPtr& input) override {
        // 1. conv -> batchnorm -> relu -> pool
        auto c = conv1->forward(input);
        auto b = bn1->forward(c);
        auto a = b->relu();
        auto p = pool->forward(a);

        // 2. flatten [B, C, H, W] to [B, C*H*W]
        int batch_size = static_cast<int>(p->shape[0]);
        auto flat = p->view({batch_size, -1});

        // 3. dense projection
        return fc->forward(flat);
    }

    void to(Device target_device) override {
        conv1->to(target_device);
        bn1->to(target_device);
        fc->to(target_device);
    }
};

void test_convnet_gpu_training() {
    std::cout << "==========================================\n";
    std::cout << "Starting GPU ConvNet End-to-End Training\n";
    std::cout << "==========================================\n";

    // Synthetic Dataset: Batch=4, Channels=1, Height=8, Width=8
    const size_t B = 4;
    const size_t C = 1;
    const size_t H = 8;
    const size_t W = 8;
    const size_t num_classes = 2;

    std::vector<double> x_vals(B * C * H * W);
    for (size_t i = 0; i < x_vals.size(); ++i) {
        x_vals[i] = std::sin(static_cast<double>(i) * 0.1);
    }

    // Binary target labels for MSE loss: [B, num_classes]
    std::vector<double> y_vals = {
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0
    };

    // Allocate input and ground truth directly on CUDA
    auto X = std::make_shared<Tensor>(x_vals, std::vector<size_t>{B, C, H, W}, false, Device::CUDA);
    auto Y = std::make_shared<Tensor>(y_vals, std::vector<size_t>{B, num_classes}, false, Device::CUDA);

    // Instantiate model and push parameters to GPU
    auto model = std::make_shared<ConvClassifier>(C, 4, num_classes);
    model->to(Device::CUDA);

    // Optimizer with momentum and adaptive moments
    Adam optimizer(model->parameters(), 0.02, {0.9, 0.999}, 1e-8, 0.001);
    MSELoss criterion;

    double initial_loss = 0.0;
    double final_loss = 0.0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < 20; ++step) {
        optimizer.zero_grad();

        auto preds = model->forward(X);
        auto loss = criterion(preds, Y);

        // Synchronize scalar loss for monitoring
        double current_loss = 0.0;
        CUDA_CHECK(cudaMemcpy(&current_loss, loss->cuda_data.get(), sizeof(double), cudaMemcpyDeviceToHost));

        if (step == 0) initial_loss = current_loss;
        if (step == 19) final_loss = current_loss;

        std::cout << "  step " << (step < 10 ? " " : "") << step 
                  << " | loss: " << current_loss << "\n";

        // Backpropagation and parameter step on GPU VRAM
        loss->backward();
        optimizer.step();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "------------------------------------------\n";
    std::cout << "  Initial Loss: " << initial_loss << " -> Final Loss: " << final_loss << "\n";
    std::cout << "  Total 20 steps elapsed: " << elapsed_ms << " ms (" << elapsed_ms / 20.0 << " ms/step)\n";

    assert(final_loss < initial_loss);
    std::cout << "[PASS] GPU ConvNet end-to-end training and convergence verified!\n";
    std::cout << "==========================================\n";
}

int main() {
    try {
        test_convnet_gpu_training();
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}