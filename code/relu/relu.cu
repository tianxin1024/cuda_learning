#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <torch/types.h>
#include <torch/extension.h>

// -------------------------------------- FP32 -------------------------------------- 
// Relu x: N, y: N y=max(0, x)
// grid(N/256), block(K=256)
__global__ void relu_f32_kernel(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}


// --------------------- PyTorch bindings for custom kernel -----------------------
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)  \
    m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                         \
if (((T).options().dtype() != (th_type))) {                          \
    std::cout << "Tensor Info:" << (T).options() << std::endl;       \
    throw std::runtime_error("values must be "#th_type);             \
}

#define TORCH_BINDING_RELU(packed_type, th_type, element_type, n_elements)      \
void relu_##packed_type(torch::Tensor x, torch::Tensor y) {                     \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                      \
    CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                      \
    const int ndim = x.dim();                                                   \
    if (ndim != 2) {                                                            \
        int N = 1;                                                              \
        for (int i = 0; i < ndim; ++i) {                                        \
            N *= x.size(i);                                                     \
        }                                                                       \
        dim3 block(256 / (n_elements));                                         \
        dim3 grid((N + 256 - 1) / 256);                                         \
        relu_##packed_type##_kernel<<<grid, block>>>(                           \
            reinterpret_cast<element_type*>(x.data_ptr()),                      \
            reinterpret_cast<element_type*>(y.data_ptr()), N);                  \
    } else {                                                                    \
        const int S = x.size(0);                                                \
        const int K = x.size(1);                                                \
        const int N = S * K;                                                    \
        if ((K / (n_elements)) <= 1024) {                                       \
            dim3 block(K / (n_elements));                                       \
            dim3 grid(S);                                                       \
            relu_##packed_type##_kernel<<<grid, block>>>(                        \
                reinterpret_cast<element_type*>(x.data_ptr()),                  \
                reinterpret_cast<element_type*>(y.data_ptr()), N);              \
        } else {                                                                \
            int N = 1;                                                          \
            for (int i = 0; i < ndim; ++i) {                                    \
                N *= x.size(i);                                                 \
            }                                                                   \
            dim3 block(256 / (n_elements));                                     \
            dim3 grid((N + 256 - 1) / 256);                                     \
            relu_##packed_type##_kernel<<<grid, block>>>(                       \
                reinterpret_cast<element_type*>(x.data_ptr()),                  \
                reinterpret_cast<element_type*>(y.data_ptr()), N);              \
        }                                                                       \
    }                                                                           \
}

TORCH_BINDING_RELU(f32, torch::kFloat32, float, 1)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(relu_f32)
}
