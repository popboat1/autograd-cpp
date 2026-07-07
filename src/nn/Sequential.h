#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "Module.h"

class Sequential : public Module {
private:
    std::vector<std::shared_ptr<Module>> layers;

public:
    void add(std::shared_ptr<Module> layer);
    
    TensorPtr forward(const TensorPtr& input) override;
    std::vector<TensorPtr> parameters() const override;
};

#endif