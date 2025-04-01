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

const size_t N = 32ULL * 1024ULL * 1024ULL;
const size_t BLOCK_SIZE = 1024;

__global__ void atomic_red(const float *gdata, float *out) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        atomicAdd(out, gdata[idx]);
    }
}

// 规约求和，使用shared memory
// grid (640, 1, 1)  block (1024, 1, 1)
__global__ void reduce_a(float *gdata, float *out) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    while (idx < N) {
        sdata[tid] += gdata[idx];
        idx += gridDim.x * blockDim.x;
    }

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

__global__ void reduce_ws(float *gdata, float *out) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    float val = 0.0f;
    unsigned mask = 0xFFFFFFFFU;

    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    while (idx < N) {
        val += gdata[idx];
        idx += gridDim.x * blockDim.x;
    }

#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (lane == 0) {
        sdata[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }

        if (tid == 0) {
            atomicAdd(out, val);
        }
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

    cudaCheck("atomic reduction kernel execution failure or cudaMemcpy H2D failure");
    if (*h_sum != (float)N) {
        printf("atomic sum reduction incorrect!\n");
    }
    printf("atomic sum reduction correct!\n");
    const int blocks = 640;
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheck("cudaMemset failure");

    reduce_a<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
    cudaCheck("reduction w/atomic kernel launch failure");

    cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheck("reduction w/atomic kernel execution failure or cudaMemcpy H2D failure");
    if (*h_sum != (float)N) {
        printf("reduction w/atomic sum incorrect!\n");
    }
    printf("reduction w/atomic sum correct!\n");
    cudaMemset(d_sum, 0, sizeof(float));
    cudaCheck("cudaMemset failure");

    reduce_ws<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
    cudaCheck("reduction warp shuffle kernel launch failure");

    cudaFree(d_A);
    cudaFree(d_sum);
    free(h_A);
    free(h_sum);

    return 0;
}
