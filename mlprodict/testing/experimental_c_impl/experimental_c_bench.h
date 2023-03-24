#pragma once

#include <cstddef>
#include <vector>
#include <cstdint>

typedef struct ElementTime {
    int64_t trial;
    int64_t row;
    double time;
    inline ElementTime() { }
    inline ElementTime(int64_t n, int64_t r, double t) { 
        trial=n;
        row=r;
        time=t; 
    }
} ElementTime;

double benchmark_cache(int64_t arr_size, bool verbose);
std::vector<ElementTime> benchmark_cache_tree(int64_t n_rows, int64_t n_features, int64_t n_trees,
                                              int64_t tree_size, int64_t max_depth, int64_t search_step=64);
