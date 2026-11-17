#ifndef DATALOADER_H
#define DATALOADER_H

#include "autograd/Tensor.h"
#include "data/Dataset.h"
#include "utils/RNG.h"
#include "utils/cuda_utils.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class DataLoader {
private:
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_;
    bool shuffle_;
    bool drop_last_;
    Device device_;

    static size_t num_elements(const std::vector<size_t>& shape) {
        size_t total = 1;
        for (size_t dim : shape) {
            total *= dim;
        }
        return total;
    }

    static std::vector<size_t> make_batch_shape(const std::vector<size_t>& sample_shape, size_t cur_bs) {
        std::vector<size_t> batch_shape;
        batch_shape.reserve(sample_shape.size() + 1);
        batch_shape.push_back(cur_bs);
        batch_shape.insert(batch_shape.end(), sample_shape.begin(), sample_shape.end());
        return batch_shape;
    }

    static void copy_sample_to_buffer(
        const TensorPtr& sample,
        const std::vector<size_t>& expected_shape,
        size_t sample_elems,
        std::vector<double>& batch_buf,
        size_t batch_offset,
        const char* field_name
    ) {
        if (!sample) {
            throw std::invalid_argument(std::string("DataLoader received null ") + field_name + " sample.");
        }
        if (sample->shape != expected_shape) {
            throw std::invalid_argument(std::string("DataLoader inconsistent ") + field_name + " sample shape in batch.");
        }
        if (sample_elems == 0) {
            return;
        }

        TensorPtr src = sample->is_contiguous() ? sample : sample->contiguous();
        double* dst = batch_buf.data() + batch_offset * sample_elems;

        if (src->device == Device::CUDA && src->cuda_data != nullptr) {
            CUDA_CHECK(cudaMemcpy(
                dst,
                src->cuda_data.get(),
                sample_elems * sizeof(double),
                cudaMemcpyDeviceToHost
            ));
        } else {
            std::copy(src->data->cbegin(), src->data->cbegin() + static_cast<std::ptrdiff_t>(sample_elems), dst);
        }
    }

    std::pair<TensorPtr, TensorPtr> collate_batch(
        const std::vector<size_t>& epoch_indices,
        size_t batch_idx
    ) const {
        const size_t start = batch_idx * batch_size_;
        const size_t cur_bs = std::min(batch_size_, epoch_indices.size() - start);

        auto first_sample = dataset_->get(epoch_indices[start]);
        if (!first_sample.first || !first_sample.second) {
            throw std::invalid_argument("DataLoader received null tensor from dataset->get().");
        }

        const std::vector<size_t> x_sample_shape = first_sample.first->shape;
        const std::vector<size_t> y_sample_shape = first_sample.second->shape;
        const size_t x_elems = num_elements(x_sample_shape);
        const size_t y_elems = num_elements(y_sample_shape);

        // stage full mini-batch on host so gpu upload takes a single h2d transfer
        std::vector<double> batch_x_buf(cur_bs * x_elems);
        std::vector<double> batch_y_buf(cur_bs * y_elems);

        copy_sample_to_buffer(first_sample.first, x_sample_shape, x_elems, batch_x_buf, 0, "x");
        copy_sample_to_buffer(first_sample.second, y_sample_shape, y_elems, batch_y_buf, 0, "y");

        for (size_t b = 1; b < cur_bs; ++b) {
            auto [sx, sy] = dataset_->get(epoch_indices[start + b]);
            copy_sample_to_buffer(sx, x_sample_shape, x_elems, batch_x_buf, b, "x");
            copy_sample_to_buffer(sy, y_sample_shape, y_elems, batch_y_buf, b, "y");
        }

        auto batch_x = std::make_shared<Tensor>(
            std::move(batch_x_buf),
            make_batch_shape(x_sample_shape, cur_bs),
            false,
            device_
        );
        auto batch_y = std::make_shared<Tensor>(
            std::move(batch_y_buf),
            make_batch_shape(y_sample_shape, cur_bs),
            false,
            device_
        );

        return {std::move(batch_x), std::move(batch_y)};
    }

public:
    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::pair<TensorPtr, TensorPtr>;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = const value_type&;

    private:
        const DataLoader* loader_ = nullptr;
        // shared per-epoch permutation keeps concurrent iterators isolated
        std::shared_ptr<const std::vector<size_t>> epoch_indices_;
        size_t batch_idx_ = 0;
        value_type current_batch_;

        void load_batch() {
            // drop previous batch tensors before allocating the next to cap peak vram
            current_batch_ = {};
            if (loader_ && epoch_indices_ && batch_idx_ < loader_->num_batches()) {
                current_batch_ = loader_->collate_batch(*epoch_indices_, batch_idx_);
            }
        }

    public:
        Iterator() = default;

        Iterator(
            const DataLoader* loader,
            std::shared_ptr<const std::vector<size_t>> epoch_indices,
            size_t batch_idx
        )
            : loader_(loader),
              epoch_indices_(std::move(epoch_indices)),
              batch_idx_(batch_idx) {
            load_batch();
        }

        reference operator*() const { return current_batch_; }
        pointer operator->() const { return &current_batch_; }

        Iterator& operator++() {
            ++batch_idx_;
            load_batch();
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator!=(const Iterator& other) const {
            return batch_idx_ != other.batch_idx_ || loader_ != other.loader_;
        }

        bool operator==(const Iterator& other) const {
            return !(*this != other);
        }
    };

    DataLoader(
        std::shared_ptr<Dataset> dataset,
        size_t batch_size,
        bool shuffle = false,
        bool drop_last = false,
        Device device = Device::CPU
    )
        : dataset_(std::move(dataset)),
          batch_size_(batch_size),
          shuffle_(shuffle),
          drop_last_(drop_last),
          device_(device) {
        if (!dataset_) {
            throw std::invalid_argument("DataLoader dataset cannot be null.");
        }
        if (batch_size_ == 0) {
            throw std::invalid_argument("DataLoader batch_size must be greater than 0.");
        }
    }

    size_t num_batches() const {
        const size_t n = dataset_->size();
        if (drop_last_) {
            return n / batch_size_;
        }
        return (n == 0) ? 0 : (n + batch_size_ - 1) / batch_size_;
    }

    Iterator begin() const {
        const size_t n = dataset_->size();
        auto epoch_indices = std::make_shared<std::vector<size_t>>(n);
        std::iota(epoch_indices->begin(), epoch_indices->end(), 0);
        if (shuffle_ && n > 1) {
            std::shuffle(epoch_indices->begin(), epoch_indices->end(), RNG::get_engine());
        }
        return Iterator(this, std::move(epoch_indices), 0);
    }

    Iterator end() const {
        return Iterator(this, nullptr, num_batches());
    }
};

#endif
