#include <stdio.h>
#include <cuda_runtime.h>

#include "common.h"

#define BLOCK_SIZE 32

void reduce_cpu(float *out, float *inp, int N) {
    for (int i = 0; i < N; i++) {
        out[0] += inp[i];
    }
}

// grid (N/block_size, 1, 1)  block (block_size, 1, 1)
__global__ void reduce_kernel_v1(float *g_odata, const float *g_idata, int N) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    while (idx < N) {
        sdata[tid] += g_idata[idx];
        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
    }
    // write result for this block to global memory
    if (tid == 0) {
        atomicAdd(g_odata, sdata[0]);
    }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

// grid (N/block_size, 1, 1)  block (block_size, 1, 1)
template <unsigned int blockSize>
__global__ void reduce_kernel_v2(float *g_odata, const float *g_idata, int N) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + tid;
    int gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;

    while (idx < N) {
        sdata[tid] += g_idata[idx] + g_idata[idx + blockSize];
        idx += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce<blockSize>(sdata, tid);
    }

    // write result for this block to global memory
    if (tid == 0) {
        atomicAdd(g_odata, sdata[0]);
    }
}

// kernel launcher
void reduce_v1(float *out, const float *inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    reduce_kernel_v1<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void reduce_v2(float *out, const float *inp, int N, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    reduce_kernel_v2<32><<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void reduce_gpu(int kernel_num, float *out, const float *inp, int N, const int block_size) {
    switch (kernel_num) {
    case 1:
        reduce_v1(out, inp, N, block_size);
        break;
    case 2:
        reduce_v2(out, inp, N, block_size);
        break;
    default:
        printf("Invalid kernel number!\n");
        exit(1);
    }
}

int main(int argc, char **argv) {
    srand(0);
    int T = 32ULL * 1024ULL * 1024ULL;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float *inp = make_random_float(T);
    float *out = (float *)malloc(sizeof(float));
    out[0] = 0.0f;

    // move to GPU
    float *d_inp, *d_out;
    cudaCheck(cudaMalloc(&d_inp, T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, T * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf(">>> kernel_num = %d\n", kernel_num);

    // int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int block_size = 32;
    if (argc > 2) {
        block_size = atoi(argv[2]);
    }
    printf(">>> block_size = %d\n", block_size);

    reduce_cpu(out, inp, T);

    // first check the correctness of the kernel and warmup
    reduce_gpu(kernel_num, d_out, d_inp, T, block_size);
    float *h_out = (float *)malloc(sizeof(float));
    validate_result(d_out, out, "out", 1, 1e-0f);

    // print the time it takes for each kernel
    int repeat_times = 100;
    float elapsed_time = benchmark_kernel(repeat_times, reduce_gpu,
                                          kernel_num, d_out, d_inp, T, block_size);

    printf("[speed] block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    printf(">>> pass test!\n");

    free(inp);
    free(out);
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_out));

    return 0;
}
