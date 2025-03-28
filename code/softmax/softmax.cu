/*
    run: nvcc -o softmax softmax.cu
*/
#include "utils.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// CPU code reference
void softmax_cpu(float *out, const float *inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float *inp_row = inp + i * C;
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }

        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }

        float norm = 1.0f / (float)sum;
        for (int j = 0; j < C; j++) {
            out_row[j] *= norm;
        }
    }
}

void softmax_online_cpu(float *out, const float *inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    for (int i = 0; i < N; i++) {
        const float *inp_row = inp + i * C;
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float maxval_prev = maxval;
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
                sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
            } else {
                sum += expf(inp_row[j] - maxval);
            }
        }

        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels
// gridDim (N / block_size, 1, 1)  blockDim (block_size, 1, 1)
__global__ void softmax_forward_kernel1(float *out, const float *inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float *inp_row = inp + i * C;
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j++) {
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
            }
        }

        double sum = 0.0;
        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }

        for (int j = 0; j < C; j++) {
            out_row[j] /= (float)sum;
        }
    }
}

// gridDim (N, 1, 1),   blockDim (block_size, 1, 1)
__global__ void softmax_forward_kernel2(float *out, const float *inp, int N, int C) {
    // inp is (N, C)
    // in each row of C elements, first calculates maxval, then returns expf(val - maxval)
    extern __shared__ float shared[];
    int idx = blockIdx.x;  // ranges [0, N)
    int tid = threadIdx.x; // ranges [0, block_size)
    int block_size = blockDim.x;
    const float *x = inp + idx * C; // idx-th row of inp
    // thread coarsening
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size) {
        maxval = fmaxf(maxval, x[i]);
    }
    shared[tid] = maxval;
    __syncthreads();

    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();
    float offset = shared[0];
    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = expf(x[i] - offset);
    }
    __syncthreads();
    // thread coarsening again, for the sum
    x = out + idx * C; // idx-th row of out
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sumval += x[i];
    }
    shared[tid] = sumval;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // broadcast the sum to all threads in the block
    __syncthreads();
    float sum = shared[0];
    // divide the input values by the sum
    for (int i = tid; i < C; i += block_size) {
        out[idx * C + i] = x[i] / sum;
    }
}

// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// warp-level reduction for finding the summing values
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// gridDim (N, 1, 1) blockDim (32, 1, 1)
__global__ void softmax_forward_kernel3(float *out, const float *inp, int N, int C) {
    // kernel must use block size of 32
    extern __shared__ float shared[];

    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = inp + idx * C;

    // Thread coarsening and within-warp reduction for maxval
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    maxval = warpReduceMax(maxval);

    // Broadcast the maxval to all threads in the block
    float offset = __shfl_sync(0xFFFFFFFF, maxval, 0);

    // Compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    // Thread coarsening and within-warp reduction for sumval
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    sumval = warpReduceSum(sumval);

    // Broadcast sumval within the warp
    sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

    // Divide the input values by the sum
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sumval;
    }
}

// gridDim (N, 1, 1), blockDim (block_size, 1, 1)
__global__ void softmax_forward_kernel4(float *out, const float *inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads
    // special reduction operations warpReduceMax/warpReduceSum are used for intra-warp reductions
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float *maxvals = shared;
    float *sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float *x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++) {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = expf(x[i] - offset);
    }

    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sumval += x[i];
    }
    sumval = warpReduceSum(sumval);

    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    if (tid == 0) {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    for (int i = tid; i < C; i += blockDim.x) {
        out[idx * C + i] = x[i] / sum;
    }
}

// gridDim (N / block_size, 1, 1)   blockDim (block_size, 1, 1)
__global__ void softmax_forward_kernel5(float *out, const float *inp, int N, int C) {
    // inp is (N, C)
    // out is (N, C), each row of inp will get softmaxed
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float *inp_row = inp + i * C;
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        double sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float maxval_prev = maxval;
            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
                sum = sum * expf(maxval_prev - maxval) + expf(inp_row[j] - maxval);
            } else {
                sum += expf(inp_row[j] - maxval);
            }
        }

        for (int j = 0; j < C; j++) {
            out_row[j] = expf(inp_row[j] - maxval) / sum;
        }
    }
}

// struct for the reduction operation, guarantees 8-byte alignment
struct __align__(8) SumMax {
    float maxval;
    float sum;
};

// forceinline helps avoid function call overhead
__device__ __forceinline__ SumMax reduce_sum_max_op(SumMax a, SumMax b) {
    bool a_bigger = (a.maxval > b.maxval);
    SumMax bigger_m = a_bigger ? a : b;
    SumMax smaller_m = a_bigger ? b : a;
    SumMax res;
    res.maxval = bigger_m.maxval;
    res.sum = bigger_m.sum + smaller_m.sum * expf(smaller_m.maxval - bigger_m.maxval);
    return res;
}

// gridDim (N * 32 / block_size, 1, 1)   blockDim (block_size, 1, 1)
__global__ void softmax_forward_kernel6(float *out, const float *inp, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

    if (idx >= N) {
        return;
    }

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float *x = inp + idx * C;

    // base case for the reduction
    SumMax sm_partial;
    sm_partial.maxval = -INFINITY;
    sm_partial.sum = 0.0f;

    // first, thread coarsening by directly accessing global memory in series
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        sm_partial = reduce_sum_max_op(sm_partial, {x[i], 1.0f});
    }

    // second, the reduction
    SumMax sm_total = cg::reduce(warp, sm_partial, reduce_sum_max_op);

    // divide the whole row by the sum
    for (int i = warp.thread_rank(); i < C; i += warp.size()) {
        // the below is equivalent to
        // out[idx * C + i] = expf(x[i] - sm_total.maxval) / sm_total.sum;
        // but uses special instruction that bypasses the cache
        __stcs(out + idx * C + i, expf(x[i] - sm_total.maxval) / sm_total.sum);
    }
}

// kernel launcher
void softmax_forward_v1(float *out, const float *inp, int N, int C, const int block_size) {
    const int grid_size = ceil_div(N, block_size);
    softmax_forward_kernel1<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward_v2(float *out, const float *inp, int N, int C, const int block_size) {
    int grid_size = N;
    size_t shared_mem_size = block_size * sizeof(float);
    softmax_forward_kernel2<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward_v3(float *out, const float *inp, int N, int C, int block_size) {
    block_size = 32;
    int grid_size = N;
    size_t shared_mem_size = block_size * sizeof(float);
    softmax_forward_kernel3<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_forward_v4(float *out, const float *inp, int N, int C, int block_size) {
    int grid_size = N;
    size_t shared_mem_size = 2 * block_size / 32 * sizeof(float);
    softmax_forward_kernel4<<<grid_size, block_size, shared_mem_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

// online softmax v1
void softmax_forward_v5(float *out, const float *inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N, block_size);
    softmax_forward_kernel5<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

// online softmax v2
void softmax_forward_v6(float *out, const float *inp, int N, int C, int block_size) {
    const int grid_size = ceil_div(N * 32, block_size);
    softmax_forward_kernel6<<<grid_size, block_size>>>(out, inp, N, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void softmax_gpu(int kernel_num, float *out, const float *inp, int N, int C, const int block_size) {
    switch (kernel_num) {
    case 1:
        softmax_forward_v1(out, inp, N, C, block_size);
        break;
    case 2:
        softmax_forward_v2(out, inp, N, C, block_size);
        break;
    case 3:
        softmax_forward_v3(out, inp, N, C, block_size);
        break;
    case 4:
        softmax_forward_v4(out, inp, N, C, block_size);
        break;
    case 5:
        softmax_forward_v5(out, inp, N, C, block_size);
        break;
    case 6:
        softmax_forward_v6(out, inp, N, C, block_size);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}

int main(int argc, char **argv) {
    srand(0);
    int B = 8;
    int T = 1024;
    int V = 4096;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float *out = (float *)malloc(B * T * V * sizeof(float));
    float *inp = make_random_float(B * T * V);

    // move to GPU
    float *d_out;
    float *d_inp;
    cudaCheck(cudaMalloc(&d_out, B * T * V * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * V * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * V * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("kernel_num : %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    softmax_cpu(out, inp, B * T, V);
    {
        float max_el = -INFINITY;
        for (int i = 0; i < B * T * V; i++) {
            max_el = max(max_el, out[i]);
        }
        assert(max_el > 1e-4);
        printf("Largest output is : %f\n", max_el);
    }

    // first check the correctness of the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        softmax_gpu(kernel_num, d_out, d_inp, B * T, V, block_size);
        validate_result(d_out, out, "out", B * T * V, 1e-4f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, softmax_gpu,
                                              kernel_num, d_out, d_inp, B * T, V, block_size);
        printf("block_size %4d | time %.4f ms | per token %.2f µs\n", block_size, elapsed_time, elapsed_time * 1000 / (B * T));
    }

    // free memory
    free(out);
    free(inp);
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));

    return 0;
}
