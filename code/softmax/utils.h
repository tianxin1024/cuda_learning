#pragma once

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>


template <class T>
T ceil_div(T dividend, T divisor) {
    return (dividend + divisor - 1) / divisor;
}

// CUDA error checking
void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// random utils
float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;  // range (-1, 1)
    }
    return arr;
}


template <class T>
void validate_result(T* device_result, const T* cpu_reference, const char* name, std::size_t num_elements, T tolerance = 1e-4) {
    T* out_gpu = (T*)malloc(num_elements * sizeof(T));
    cudaCheck(cudaMemcpy(out_gpu, device_result, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    int nfaults = 0;
    for (int i = 0; i < num_elements; i++) {
        // print the first few comparisons
        if (i < 1) {
            printf("%f %f\n", cpu_reference[i], out_gpu[i]);
        }
        // ensure correctness for all elements
        if (fabs(cpu_reference[i] - out_gpu[i]) > tolerance) {
            printf("Mismatch of %s at %d: CPU_ref: %f vs GPU: %f\n", name, i, cpu_reference[i], out_gpu[i]);
            nfaults++;
            if (nfaults >= 10) {
                free(out_gpu);
                exit(EXIT_FAILURE);
            }
        }
    }

    free(out_gpu);
}


template <class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(start, nullptr));
    for (int i = 0; i < repeats; i++) {
        kernel(std::forward<KernelArgs>(kernel_args)...);
    }
    cudaCheck(cudaEventRecord(stop, nullptr));
    cudaCheck(cudaEventSynchronize(start));
    cudaCheck(cudaEventSynchronize(stop));
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));

    return elapsed_time / repeats;
}
