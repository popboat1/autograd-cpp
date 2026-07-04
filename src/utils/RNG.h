#ifndef RNG_H
#define RNG_H

#include <random>

class RNG {
public:
    // returns a static reference to a single global execution engine
    static std::mt19937& get_engine() {
        static std::mt19937 engine(42); // default seed
        return engine;
    }

    // call this anywhere to re-seed the entire global state (similar to torch.manual_seed)
    static void manual_seed(unsigned int seed) {
        get_engine().seed(seed);
    }
};

#endif