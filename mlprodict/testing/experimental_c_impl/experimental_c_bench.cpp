#include "experimental_c_bench.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdexcept>

// source: https://stackoverflow.com/questions/9412585/see-the-cache-missess-simple-c-cache-benchmark

long get_nsec() {
   timespec ts;
   clock_gettime(CLOCK_REALTIME, &ts);
   return long(ts.tv_sec)*1000*1000 + ts.tv_nsec;
}


float benchmark_cache(const size_t arr_size, bool verbose) {

    unsigned char* arr_a = (unsigned char*) malloc(sizeof(char) * arr_size);
    unsigned char* arr_b = (unsigned char*) malloc(sizeof(char) * arr_size);

    if (arr_a == nullptr || arr_b == nullptr)
        throw std::runtime_error("An array could not be allocated.");

    long time0 = get_nsec();

    for(size_t i = 0; i < arr_size; ++i) {
        // index k will jump forth and back, to generate cache misses
        size_t k = (i / 2) + (i % 2) * arr_size / 2;
        arr_b[k] = arr_a[k] + 1;
    }

    long time_d = get_nsec() - time0;
    float performance = float(time_d) / arr_size;
    if (verbose) {
        printf("perf %.1f [kB]: %ld\n", performance, arr_size / 1024 );
    }

    free(arr_a);
    free(arr_b);
    return performance;
}


