#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "utils.h"

void layernorm_cpu(float *out, float *mean, float *rstd,
                   const float *inp, const float *weight, const float *bias, int B, int T, int C) {
    float eps = 1e-5;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the inpu position inp[b, t, :]
            const float *x = inp + b * T * C + t * C;
            // calculate mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m / C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            // calculate the rstd
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float *out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m));        // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o;                     // write
            }
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

// GPU Kernel
// gridDim (N/block_size, 1, 1)   blockDim (block_size, 1, 1)
__global__ void layernorm_kernel_v1(float *out, float *mean, float *rstd,
                                    const float *inp, const float *weight, const float *bias,
                                    int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if (idx > N) {
        return;
    }

    // seek to the input position inp[idx, :]
    const float *x = inp + idx * C;
    // calcuate the mean
    float m = 0.0f;
    for (int i = 0; i < C; i++) {
        m += x[i];
    }
    m = m / C;
    // calcuate the variance (without any bias correction)
    float v = 0.0f;
    for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
    }
    v = v / C;
    // calcuate the rstd
    float s = 1.0f / sqrtf(v + eps);
    // seek to the output position in out[idx, :]
    float *out_idx = out + idx * C;
    for (int i = 0; i < C; i++) {
        float n = (s * (x[i] - m));
        float o = n * weight[i] + bias[i];
        out_idx[i] = o;
    }
    mean[idx] = m;
    rstd[idx] = s;
}

// gridDim (N, 1, 1), blockDim (block_size, 1, 1)
__global__ void mean_kernel(float *mean, const float *inp, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x;  // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)

    const float *x = inp + idx * C;
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }

    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}

// gridDim (N, 1, 1)  blockDim (block_size, 1, 1)
__global__ void rstd_kernel(float *rstd, const float *inp, const float *mean, int N, int C, int block_size) {
    extern __shared__ float shared[];
    int idx = blockIdx.x;  // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)

    const float *x = inp + idx * C;
    float m = mean[idx];
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        float xshift = x[i] - m;
        sum += xshift * xshift;
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
    }
}

// gridDim (B*T*C / 256, 1, 1)  blockDim (256, 1, 1)
__global__ void normalization_kernel(float *out, const float *inp, float *mean, float *rstd,
                                     const float *weight, const float *bias, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int bt = idx / C;
    int c = idx % C;

    float m = mean[bt];
    float s = rstd[bt];
    float xi = inp[idx];
    float n = s * (xi - m);
    float o = n * weight[c] + bias[c];

    out[idx] = o;
}

// gridDim (N *32 / block_size, 1, 1)  blockDim (block_size, 1, 1)
__global__ void layernorm_kernel_v3(float *__restrict__ out, float *__restrict__ mean, float *__restrict__ rstd,
                                    const float *__restrict__ inp, const float *__restrict__ weight,
                                    const float *__restrict__ bias, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // meta_group_size is the number of warps in a block, and meta_group_rank is the warp index
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N) {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float *x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sum += x[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;
    if (warp.thread_rank() == 0 && mean != nullptr) {
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if (warp.thread_rank() == 0 && rstd != nullptr) {
        __stcs(rstd + idx, s);
    }

    // final normalization and scaling by weight/bias
    float *o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size()) {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * (__ldcs(x + c) - m);
        __stcs(o + c, n * weight[c] + bias[c]);
    }
}

// Kernel launcher
void layernorm_forward_v1(float *out, float *mean, float *rstd,
                          const float *inp, const float *weight, const float *bias,
                          int B, int T, int C,
                          const int block_size) {
    const int N = B * T;
    const int grid_size = ceil_div(N, block_size);
    layernorm_kernel_v1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward_v2(float *out, float *mean, float *rstd,
                          const float *inp, const float *weight, const float *bias,
                          int B, int T, int C,
                          const int block_size) {
    int N = B * T;
    // in mean and rstd, threads cooperate within blocks via reductions
    mean_kernel<<<N, block_size, block_size * sizeof(float)>>>(mean, inp, N, C, block_size);
    cudaCheck(cudaGetLastError());
    rstd_kernel<<<N, block_size, block_size * sizeof(float)>>>(rstd, inp, mean, N, C, block_size);
    cudaCheck(cudaGetLastError());
    // in the normalization, everything just gets flattened out
    const int block_size2 = 256;
    const int grid_size = ceil_div(B * T * C, block_size2);
    normalization_kernel<<<grid_size, block_size2>>>(out, inp, mean, rstd, weight, bias, B, T, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward_v3(float *out, float *mean, float *rstd,
                          const float *inp, const float *weight, const float *bias,
                          int B, int T, int C,
                          const int block_size) {
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = ceil_div(N * 32, block_size);
    layernorm_kernel_v3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// Kernel version dispatch
void layernorm_gpu(int kernel_num,
                   float *out, float *mean, float *rstd,
                   const float *inp, const float *weight, const float *bias,
                   int B, int T, int C,
                   const int block_size) {
    switch (kernel_num) {
    case 1:
        layernorm_forward_v1(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
        break;
    case 2:
        layernorm_forward_v2(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
        break;
    case 3:
        layernorm_forward_v3(out, mean, rstd, inp, weight, bias, B, T, C, block_size);
        break;
    default:
        printf("Invalid kernel_num %d\n", kernel_num);
        exit(1);
    }
}

int main(int argc, char **argv) {
    srand(0);

    int B = 8;
    int T = 1024; // seq_lens
    int C = 768;  // dim models

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float *out = (float *)malloc(B * T * C * sizeof(float));
    float *mean = (float *)malloc(B * T * sizeof(float));
    float *rstd = (float *)malloc(B * T * sizeof(float));
    float *inp = make_random_float(B * T * C);
    float *weight = make_random_float(C);
    float *bias = make_random_float(C);

    // move to GPU
    float *d_out;
    float *d_mean;
    float *d_rstd;
    float *d_inp;
    float *d_weight;
    float *d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    layernorm_cpu(out, mean, rstd, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        layernorm_gpu(kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias, B, T, C, block_size);

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
        validate_result(d_mean, mean, "mean", B * T, 1e-5f);
        validate_result(d_rstd, rstd, "rstd", B * T, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm_gpu,
                                              kernel_num, d_out, d_mean, d_rstd, d_inp, d_weight, d_bias,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(mean);
    free(rstd);
    free(inp);
    free(weight);
    free(bias);

    return 0;
}
