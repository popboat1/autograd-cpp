#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "Module.h"

class Sequential : public Module {
private:
    // preserves sequential execution order independently from extra submodules
    std::vector<std::shared_ptr<Module>> layers;

public:
    void add(std::shared_ptr<Module> layer);

    TensorPtr forward(const TensorPtr& input) override;
};

#endif