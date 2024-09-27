#include "relu.cuh"

// -------------------------------------- FP32 -------------------------------------- 
// Relu x: N, y: N y=max(0, x)
// grid(N/256), block(K=256)
__global__ void relu_f32_kernel(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

// Relu x: N, y: N y=max(0,x) Vec4
// grid(N/256/4), block(256/4)
__global__ void relu_f32x4_kernel(float* x, float* y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = fmaxf(0.0f, reg_x.x);
        reg_y.y = fmaxf(0.0f, reg_x.y);
        reg_y.z = fmaxf(0.0f, reg_x.z);
        reg_y.w = fmaxf(0.0f, reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

// -------------------------------------- FP16 -------------------------------------- 
__global__ void relu_f16_kernel(half* x, half* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = __hmax(__float2half(0.0f), x[idx]);
    }
}

__global__ void relu_f16x2_kernel(half* x, half* y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx < N) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y = HALF2(y[idx]);
        reg_y.x = __hmax(__float2half(0.0f), reg_x.x);
        reg_y.y = __hmax(__float2half(0.0f), reg_x.y);
        HALF2(y[idx]) = reg_y;
    }
}

