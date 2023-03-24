#pragma once

#include <cstddef>
#include <vector>

typedef struct elem_time {
    int64_t n_trial;
    int64_t row;
    double time;
    inline elem_time() { }
    inline elem_time(int64_t n, int64_t r, double t) { 
        n_trial=t; row=r; time=r; 
    }
} elem_time;

double benchmark_cache(int64_t arr_size, bool verbose);
std::vector<elem_time> benchmark_cache_tree(
        int64_t n_rows, int64_t n_features, int64_t n_trees, int64_t tree_size,
        int64_t max_depth, int64_t n_trials);
