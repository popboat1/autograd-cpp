#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

// runtime assertions that evaluate unconditionally regardless of NDEBUG
#define CHECK_TRUE(cond) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error( \
                std::string("[FAIL] Assertion failed: (") + #cond + ") at " + \
                __FILE__ + ":" + std::to_string(__LINE__) \
            ); \
        } \
    } while (0)

#define CHECK_FALSE(cond) CHECK_TRUE(!(cond))

// custom macro for tensor invariant and condition checks
#define CHECK_TENSOR(cond) CHECK_TRUE(cond)

#define CHECK_EQUAL(a, b) \
    do { \
        if (!((a) == (b))) { \
            throw std::runtime_error( \
                std::string("[FAIL] Equality mismatch: (") + #a + " == " + #b + ") failed at " + \
                __FILE__ + ":" + std::to_string(__LINE__) \
            ); \
        } \
    } while (0)

// runtime floating-point parity assertion with error diff display
#define CHECK_CLOSE(a, b, tol) \
    do { \
        double _diff = std::abs(static_cast<double>(a) - static_cast<double>(b)); \
        if (_diff > (tol)) { \
            throw std::runtime_error( \
                std::string("[FAIL] Parity mismatch: |") + #a + " - " + #b + \
                "| = " + std::to_string(_diff) + " > tol " + std::to_string(tol) + \
                " (" + std::to_string(a) + " vs " + std::to_string(b) + ") at " + \
                __FILE__ + ":" + std::to_string(__LINE__) \
            ); \
        } \
    } while (0)

// CUDA runtime API error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("[CUDA FAIL] ") + cudaGetErrorString(_err) + " (" + #call + ") at " + \
                __FILE__ + ":" + std::to_string(__LINE__) \
            ); \
        } \
    } while (0)

// backward compatibility aliases
#define ASSERT_TRUE(cond) CHECK_TRUE(cond)
#define ASSERT_FALSE(cond) CHECK_FALSE(cond)
#define ASSERT_CLOSE(a, b, tol) CHECK_CLOSE(a, b, tol)

// override standard assert to ensure all existing test calls run in Release builds
#ifdef assert
#undef assert
#endif
#define assert(cond) CHECK_TRUE(cond)

#endif // TEST_UTILS_H