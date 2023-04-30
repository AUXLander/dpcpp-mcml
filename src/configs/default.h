#ifndef MCML_CONFIG
#define MCML_CONFIG

constexpr size_t LAYER_OUTPUT_SIZE_X = 16;
constexpr size_t LAYER_OUTPUT_SIZE_Y = 16;
constexpr size_t LAYER_OUTPUT_SIZE_Z = 16;
constexpr size_t LAYER_OUTPUT_COUNT  = 1;

constexpr size_t SIMULATION_RANDOM_SEED   = 42;
constexpr size_t SIMULATION_LAYERS_COUNT  = 24;
constexpr size_t SIMULATION_REPEATS_COUNT = 1024; // 128;

constexpr size_t CONFIGURATION_WORK_GROUP_COUNT = 256;
constexpr size_t CONFIGURATION_WORK_GROUP_SIZE  = 256;

// #define FEATURE_USE_GROUP_SUMMATOR
// #define FEATURE_USE_LOCAL_MEMORY
// #define FEATURE_USE_ATOMIC_SUMMATOR

#endif