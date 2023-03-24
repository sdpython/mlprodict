#include "experimental_c_bench.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdexcept>
#include <chrono>
#include <iostream>

// source: https://stackoverflow.com/questions/9412585/
// see-the-cache-missess-simple-c-cache-benchmark


double benchmark_cache(int64_t arr_size, bool verbose) {

    unsigned char* arr_a = (unsigned char*) malloc(sizeof(char) * arr_size);
    unsigned char* arr_b = (unsigned char*) malloc(sizeof(char) * arr_size);

    if (arr_a == nullptr || arr_b == nullptr)
        throw std::runtime_error("An array could not be allocated.");

    auto time0 = std::chrono::high_resolution_clock::now();

    for(int64_t i = 0; i < arr_size; ++i) {
        // index k will jump forth and back, to generate cache misses
        int64_t k = (i / 2) + (i % 2) * arr_size / 2;
        arr_b[k] = arr_a[k] + 1;
    }

    double performance = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - time0).count();
    performance /= arr_size;
    if (verbose) {
        printf("perf %.1f [kB]: %d\n", performance, (int)(arr_size / 1024));
    }

    free(arr_a);
    free(arr_b);
    return performance;
}

std::vector<elem_time> benchmark_cache_tree(
        int64_t n_rows, int64_t n_features, int64_t n_trees, int64_t tree_size,
        int64_t max_depth, int64_t n_trials) {
    
    std::vector<float> X(n_rows * n_features);
    for(int64_t i=0; i < X.size(); ++i) X[i] = (float)i / (float)X.size();
    std::vector<float> T(n_trees * tree_size);
    for(int64_t i=0; i < X.size(); ++i) T[i] = (float)i / (float)T.size();
    std::vector<float> res(n_trees * n_rows, 0);
            
    // std::cout << "X.size()=" << X.size() << " T.size()=" << T.size()
    //           << " res.size()=" << res.size() << "\n";
    
    int64_t seed = n_features * 7 + 1;
    int64_t fi, ti;
    std::vector<elem_time> times(n_rows * n_trials);
    for(int64_t n_trial=0; n_trial < n_trials; ++n_trial) {
        for(int64_t i=0; i < n_rows; i += 1) {
            // std::cout << "i=" << i << "\n";
            auto time0 = std::chrono::high_resolution_clock::now();
            for (int64_t t=0; t < n_trees; ++t, ++seed) {
                if (seed > 10037) seed = n_features * 7 + 1;
                for (int64_t mx=0; mx < max_depth; ++mx) {
                    fi = i * n_features + ((seed * (t + mx)) % n_features);
                    ti = t * tree_size + ((seed * (i + mx)) % tree_size);
                    res[i * n_trees + t] += X[fi] - T[ti];
                }
            }
            double performance = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - time0).count();            
            times[n_trial * n_rows + i] = elem_time(n_trial, i, performance);
        }
    }
    return times;
}


