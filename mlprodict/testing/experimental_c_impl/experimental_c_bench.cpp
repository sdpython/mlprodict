#include "experimental_c_bench.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <cstring>

#if USE_OPENMP
#include <omp.h>
#endif

// source: https://stackoverflow.com/questions/9412585/
// see-the-cache-missess-simple-c-cache-benchmark


double benchmark_cache(int64_t arr_size, bool verbose) {

    unsigned char* arr_a = (unsigned char*) malloc(sizeof(char) * arr_size);
    unsigned char* arr_b = (unsigned char*) malloc(sizeof(char) * arr_size);
    memset(arr_a, 1, sizeof(char) * arr_size);

    if (arr_a == nullptr || arr_b == nullptr)
        throw std::runtime_error("An array could not be allocated.");

    // do the real measure
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

std::vector<ElementTime> benchmark_cache_tree(
        int64_t n_rows, int64_t n_features, int64_t n_trees,
        int64_t tree_size, int64_t max_depth, int64_t search_step) {
    
    std::vector<float> X(n_rows * n_features);
    for(int64_t i=0; i < static_cast<int64_t>(X.size()); ++i)
        X[i] = (float)i / (float)X.size();

    std::vector<float> T(n_trees * tree_size);
    for(int64_t i=0; i < static_cast<int64_t>(T.size()); ++i)
        T[i] = (float)i / (float)T.size();

    std::vector<float> res(n_trees * n_rows, 0);
    
    int64_t seed = n_features * 7 + 1;
    int64_t fi, ti;
    std::vector<ElementTime> times(n_rows);
    for(int64_t step=search_step; step < n_rows; step += search_step) {
        auto time0 = std::chrono::high_resolution_clock::now();

        for(int64_t batch=0; batch < n_rows; batch += step, ++seed) {
            if (seed > 10037) seed = n_features * 7 + 1;
            #if USE_OPENMP
            #pragma omp parallel for
            #endif
            for (int64_t t=0; t < n_trees; ++t) {                
                int64_t end = batch + step < n_rows ? batch + step : n_rows;
                for (int64_t i = batch; i < end; ++i) {
                    for (int64_t mx=0; mx < max_depth; ++mx) {
                        fi = i * n_features + ((seed * (t + mx)) % n_features);
                        ti = t * tree_size + ((seed * (i + mx)) % tree_size);
                        res[i * n_trees + t] += X[fi] - T[ti];
                    }
                }
            }
        }

        double performance = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - time0).count();
        if (step == search_step) {
            // first iteration
            for(int64_t i=0; i < static_cast<int64_t>(times.size()); ++i) {
                times[i] = ElementTime(0, i, performance);
            }
        }
        for(int64_t i=step - search_step; i < step; ++i) {
            times[i] = ElementTime(0, i, performance);
        }
    }
    return times;
}


