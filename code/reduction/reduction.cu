#include <stdio.h>
#include <cuda_runtime.h>

#define cudaCheck(msg)                                                   \
    do {                                                                 \
        cudaError_t __err = cudaGetLastError();                          \
        if (__err != cudaSuccess) {                                      \
            fprintf(stderr, "False error: %s (%s at %s:%d)\n",           \
                    msg, cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** Failed - Aborting\n");                  \
        }                                                                \
    } while (0)

const size_t N = 33ULL * 1024ULL * 1024ULL;
const size_t BLOCK_SIZE = 1024;

__global__ void atomic_red(const float *gdata, float *out) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        atomicAdd(out, gdata[idx]);
    }
}

int main() {
    float *h_A, *h_sum, *d_A, *d_sum;
    h_A = new float[N];
    h_sum = new float;
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
    }
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    cudaCheck("cudaMalloc failure");

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheck("cudaMemcpy H2D failure");

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheck("cudaMemset failure");

    atomic_red<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
    cudaCheck("atomic reduction kernel launch failure");
    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_sum);
    free(h_A);
    free(h_sum);

    return 0;
}
