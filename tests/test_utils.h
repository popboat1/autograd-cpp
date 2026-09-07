#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>

// runtime assertion that throws regardless of NDEBUG
#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error( \
                std::string("[FAIL] Assertion failed: (") + #cond + ") at " + \
                __FILE__ + ":" + std::to_string(__LINE__) \
            ); \
        } \
    } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

// runtime floating-point parity assertion with error diff display
#define ASSERT_CLOSE(a, b, tol) \
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

// override standard assert to ensure all existing test calls run in Release builds
#ifdef assert
#undef assert
#endif
#define assert(cond) ASSERT_TRUE(cond)

#endif // TEST_UTILS_H