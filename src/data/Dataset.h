#ifndef DATASET_H
#define DATASET_H

#include "autograd/Tensor.h"
#include "utils/cuda_utils.h"

#include <cstddef>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

class Dataset
{
public:
    virtual ~Dataset() = default;
    virtual size_t size() const = 0;
    virtual std::pair<TensorPtr, TensorPtr> get(size_t index) const = 0;
};

class TensorDataset : public Dataset
{
private:
    TensorPtr data_;
    TensorPtr targets_;

    // cached metadata
    size_t num_samples;
    size_t x_sample_size;
    size_t y_sample_size;
    std::vector<size_t> x_sample_shape;
    std::vector<size_t> y_sample_shape;

    static TensorPtr normalize_host_tensor(const TensorPtr& t, const char* name)
    {
        if(!t || t->shape.empty()){
            throw std::invalid_argument(std::string(name) + " must be non-null with at least 1 dimension.");
        }

        TensorPtr x = t->is_contiguous() ? t : t->contiguous();

        if(x->device == Device::CUDA && x->cuda_data != nullptr)
        {   
            std::vector<double> host_vals(x->data->size());

            CUDA_CHECK(cudaMemcpy(
                host_vals.data(),
                x->cuda_data.get(),
                host_vals.size() * sizeof(double),
                cudaMemcpyDeviceToHost
            ));

            return std::make_shared<Tensor>(std::move(host_vals), x->shape, false, Device::CPU);
        }

        return x;
    }

    static TensorPtr slice_sample(
        const TensorPtr& src,
        size_t index,
        size_t sample_size,
        const std::vector<size_t>& sample_shape
    )
    {
        size_t offset = index * sample_size;

        auto begin = src->data->cbegin() + static_cast<std::ptrdiff_t>(offset);
        std::vector<double> vals(begin, begin + static_cast<std::ptrdiff_t>(sample_size));

        return std::make_shared<Tensor>(std::move(vals), sample_shape, false, Device::CPU);
    }

public:
    // constructor
    TensorDataset(TensorPtr data, TensorPtr targets){
        data_ = normalize_host_tensor(data, "data");
        targets_ = normalize_host_tensor(targets, "targets");

        // validate matching sample counts
        if (data_->shape[0] != targets_->shape[0])
        {
            throw std::invalid_argument(
                "TensorDataset sample count mismatch: data has " +
                std::to_string(data_->shape[0]) + " samples, targets has " +
                std::to_string(targets_->shape[0]) + "."
            );
        }

        // cache metadata
        num_samples = data_->shape[0];
        x_sample_size = data_->strides[0];
        y_sample_size = targets_->strides[0];
        x_sample_shape.assign(data_->shape.begin() + 1, data_->shape.end());
        y_sample_shape.assign(targets_->shape.begin() + 1, targets_->shape.end());
    }

    size_t size() const override { return num_samples; }

    std::pair<TensorPtr, TensorPtr> get(size_t index) const override
    {
        if(index >= num_samples)
        {
            throw std::out_of_range("TensorDataset index out of bounds: " 
                + std::to_string(index) + " >= " 
                + std::to_string(num_samples) + "!");
        }

        return { 
            slice_sample(data_, index, x_sample_size, x_sample_shape), 
            slice_sample(targets_, index, y_sample_size, y_sample_shape)
        };
    }
};


#endif