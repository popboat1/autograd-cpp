#include "Dropout.h"
#include "utils/RNG.h"
#include <cstdint>
#include <stdexcept>
#include "utils/cuda_utils.h"

Dropout::Dropout(double p) : p(p)
{
    if(!(p >= 0.0 && p < 1.0))
    {
        throw std::invalid_argument("dropout p value must be between 0.0 and 1.0 [0.0, 1.0)!");
    }
}

// ----- kernels for Dropout
__host__ __device__ inline double uniform_01(uint64_t seed, size_t idx) {
    uint64_t z = seed + static_cast<uint64_t>(idx) * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0); // / 2^53
}

__global__ void d_dropout_fwd(
    const double* __restrict__ in,
    double* __restrict__ out,
    double* __restrict__ mask,
    size_t n,
    double p,
    double scale,
    uint64_t seed
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    double m = (uniform_01(seed, idx) >= p) ? scale : 0.0;
    mask[idx] = m;
    out[idx] = in[idx] * m;
}

__global__ void d_dropout_bwd(
    const double* __restrict__ upstream_grad,
    double* __restrict__ in_grad,
    const double* __restrict__ mask,
    size_t n
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= n) return;

    in_grad[idx] += upstream_grad[idx] * mask[idx];
}

// Dropout forward
TensorPtr Dropout::forward(const TensorPtr& input) {
    if (!this->training || p == 0.0) {
        return input;
    }

    TensorPtr x = input->is_contiguous() ? input : input->contiguous();
    size_t total_elements = x->data->size();
    if (total_elements == 0) return x;

    double scale = 1.0 / (1.0 - p);

    auto& gen = RNG::get_engine();
    uint64_t hi = static_cast<uint64_t>(gen());
    uint64_t lo = static_cast<uint64_t>(gen());
    uint64_t seed = (hi << 32) | lo;

    if(x->device == Device::CUDA)
    {
        auto out = std::make_shared<Tensor>(
            std::vector<double>(total_elements, 0.0),
            x->shape,
            std::vector<TensorPtr>{x},
            "dropout",
            Device::CUDA
        );
    
        double* raw_mask = nullptr;
        CUDA_CHECK(cudaMalloc(&raw_mask, total_elements * sizeof(double)));
        auto d_mask = std::shared_ptr<double>(raw_mask, cuda_utils::CudaDeleter{});

        constexpr int block_threads = 256;
        int blocks = cuda_utils::ceil_div(static_cast<int>(total_elements), block_threads);
        d_dropout_fwd<<<blocks, block_threads>>>(
            x->cuda_data.get(), out->cuda_data.get(), d_mask.get(),
            total_elements, p, scale, seed
        );
        CUDA_CHECK(cudaGetLastError());

        std::weak_ptr<Tensor> weak_out = out;
        std::weak_ptr<Tensor> weak_x = x;
        out->backward_func = [weak_x, weak_out, total_elements, d_mask, blocks, block_threads]() {
            auto out_ptr = weak_out.lock();
            auto x_ptr = weak_x.lock();
            if(out_ptr && x_ptr && out_ptr->requires_grad && x_ptr->requires_grad)
            {
                x_ptr->ensure_grad_allocated();
                d_dropout_bwd<<<blocks, block_threads>>>(
                    out_ptr->cuda_grad.get(), x_ptr->cuda_grad.get(), 
                    d_mask.get(), total_elements
                );
                CUDA_CHECK(cudaGetLastError());
            }
        };

        return out;

    }  else {
        // cpu path
        auto mask = std::make_shared<std::vector<double>>(total_elements, 0.0);
        std::vector<double> out_vals(total_elements, 0.0);

        #pragma omp parallel for
        for(size_t i = 0; i < total_elements; ++i)
        {
            double m {(uniform_01(seed, i) >= p) ? scale : 0.0};
            (*mask)[i] = m;
            out_vals[i] = (*x->data)[i] * m;
        }

        auto out = std::make_shared<Tensor>(std::move(out_vals), x->shape, std::vector<TensorPtr>{x}, "dropout", Device::CPU);

        std::weak_ptr<Tensor> weak_out = out;
        std::weak_ptr<Tensor> weak_x = x;
        out->backward_func = [weak_x, weak_out, total_elements, mask]() {
            auto out_ptr = weak_out.lock();
            auto x_ptr = weak_x.lock();
            if(out_ptr && x_ptr && out_ptr->requires_grad && x_ptr->requires_grad)
            {
                x_ptr->ensure_grad_allocated();

                #pragma omp parallel for
                for(size_t i = 0; i < total_elements; ++i)
                {
                    (*x_ptr->grad)[i] += (*out_ptr->grad)[i] * (*mask)[i];
                }
            }
        };

        return out;
    }
}


// ------- Dropout2D

// --- kernels

__global__ void d_dropout2d_fwd(
    const double* __restrict__ in,
    double* __restrict__ out,
    double* __restrict__ channel_mask,
    size_t total_elements,
    size_t spatial_size,
    double p,
    double scale,
    uint64_t seed
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_elements) return;

    size_t bc = idx / spatial_size;
    double m = (uniform_01(seed, bc) >= p) ? scale : 0.0;

    out[idx] = in[idx] * m;

    if (idx % spatial_size == 0){
        channel_mask[bc] = m;
    }
}


__global__ void d_dropout2d_bwd(
    const double* __restrict__ upstream_grad,
    double* __restrict__ in_grad,
    const double* __restrict__ channel_mask,
    size_t total_elements,
    size_t spatial_size
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_elements) return;

    size_t bc = idx / spatial_size;

    // accumulate
    in_grad[idx] += upstream_grad[idx] * channel_mask[bc]; // no atomic add needed since each idx map 1 to 1
}

// ---

Dropout2D::Dropout2D(double p) : p(p)
{
    if(!(p >= 0.0 && p < 1.0))
    {
        throw std::invalid_argument("dropout p value must be between 0.0 and 1.0 [0.0, 1.0)!");
    }
}


TensorPtr Dropout2D::forward(const TensorPtr& input) {
    if (input->shape.size() != 4) {
        throw std::invalid_argument("Dropout2D expects a 4D NCHW input tensor!");
    }
    if (!this->training || p == 0.0) {
        return input;
    }

    // normalise contiguity
    TensorPtr x = input->is_contiguous() ? input : input->contiguous();

    // unpack
    size_t B = x->shape[0];
    size_t C = x->shape[1];
    size_t H = x->shape[2];
    size_t W = x->shape[3];

    size_t spatial_size = H * W;
    size_t num_planes = B * C;
    size_t total_elements = num_planes * spatial_size;

    if (total_elements == 0) return x;

    auto& gen = RNG::get_engine();
    uint64_t hi = static_cast<uint64_t>(gen());
    uint64_t lo = static_cast<uint64_t>(gen());
    uint64_t seed = (hi << 32) | lo;

    double scale = 1.0 / (1.0 - p);

    if(x->device == Device::CUDA)
    {
        auto out = std::make_shared<Tensor>(
            std::vector<double>(total_elements, 0.0),
            x->shape,
            std::vector<TensorPtr>{x},
            "dropout2d",
            Device::CUDA
        );
        
        double* raw_mask = nullptr;
        CUDA_CHECK(cudaMalloc(&raw_mask, num_planes * sizeof(double)));
        auto d_channel_mask = std::shared_ptr<double>(raw_mask, cuda_utils::CudaDeleter{});

        // launch forward kernel
        constexpr int block_threads = 256;
        int blocks = cuda_utils::ceil_div(static_cast<int>(total_elements), block_threads);
        d_dropout2d_fwd<<<blocks, block_threads>>>(
            x->cuda_data.get(), out->cuda_data.get(), d_channel_mask.get(),
            total_elements, spatial_size, p, scale, seed
        );
        CUDA_CHECK(cudaGetLastError());


        // backward pass on GPU
        std::weak_ptr<Tensor> weak_out = out;
        std::weak_ptr<Tensor> weak_x = x;
        out->backward_func = [
            weak_x, weak_out, total_elements, d_channel_mask, 
            spatial_size, blocks, block_threads](){
            
            auto out_ptr = weak_out.lock();
            auto x_ptr = weak_x.lock();
            if(out_ptr && x_ptr && out_ptr->requires_grad && x_ptr->requires_grad)
            {
                x_ptr->ensure_grad_allocated();
                d_dropout2d_bwd<<<blocks, block_threads>>>(
                    out_ptr->cuda_grad.get(), x_ptr->cuda_grad.get(), d_channel_mask.get(),
                    total_elements, spatial_size
                );
                CUDA_CHECK(cudaGetLastError());
            }
        };

        return out;

    } else
    // cpu path
    {
        auto channel_mask = std::make_shared<std::vector<double>>(num_planes, 0.0);
        std::vector<double> out_vals(total_elements, 0.0);

        // prefill
        for (size_t bc = 0; bc < num_planes; ++bc){
            (*channel_mask)[bc] = (uniform_01(seed, bc) >= p) ? scale : 0.0;
        }

        #pragma omp parallel for
        for(int i{}; i < total_elements; ++i)
        {
            out_vals[i] = (*x->data)[i] * (*channel_mask)[i / spatial_size];
        }

        auto out = std::make_shared<Tensor>(
            std::move(out_vals), 
            x->shape, 
            std::vector<TensorPtr>{x}, 
            "dropout2d", 
            Device::CPU
        );


        // backward pass on cpu
        std::weak_ptr<Tensor> weak_out = out;
        std::weak_ptr<Tensor> weak_x = x;
        out->backward_func = [weak_x, weak_out, total_elements, channel_mask, spatial_size](){
            auto out_ptr = weak_out.lock();
            auto x_ptr = weak_x.lock();
            if(out_ptr && x_ptr && out_ptr->requires_grad && x_ptr->requires_grad)
            {
                x_ptr->ensure_grad_allocated();

                #pragma omp parallel for
                for(int i{}; i < total_elements; ++i)
                {
                    (*x_ptr->grad)[i] += (*out_ptr->grad)[i] * (*channel_mask)[i / spatial_size];
                }
            }
        };

        return out;
    }
}


