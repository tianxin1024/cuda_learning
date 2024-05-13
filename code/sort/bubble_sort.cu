#include <iostream>
#include "cuda_runtime.h"

template<typename T>
struct JudgeSwap {
    __host__ __device__
    virtual bool operator() (const T left, const T right) const;
};

template<typename T>
__host__ __device__ __inline__
void swap(T* a, T* b);

template<typename T>
__global__
void bubbleSort(T* v, const unsigned int n, JudgeSwap<T> is_swap);


template<typename T>
__host__ __device__
bool JudgeSwap<T>::operator() (const T left, const T right) const {
    return left > right;
}

template<typename T>
__host__ __device__ __inline__
void swap(T* a, T* b) {
    T tmp = *a;
    *a = *b;
    *b = tmp;
}

template<typename T>
__global__
void bubbleSort(T* v, const unsigned int n, JudgeSwap<T> is_swap) {
    const unsigned int tid = threadIdx.x;

    for (unsigned int i = 0; i < n; ++i) {
        unsigned int offset = i % 2;
        unsigned int curr_index = 2 * tid + offset;
        unsigned int next_index = curr_index + 1;

        if (next_index < n) {
            if (is_swap(v[curr_index], v[next_index])) {
                swap<T>(&v[curr_index], &v[next_index]);
            }
        }
        __syncthreads();
    }
}

int main() {

    int size;
    std::cout << "Enter the size of the array: ";
    std::cin >> size;

    int* h_v;

    h_v = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; ++i) {
        h_v[i] = rand() % 100;
    }

    int *d_v;
    cudaMalloc((void**)&d_v, size * sizeof(int));
    cudaMemcpy(d_v, h_v, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grdDim(1, 1, 1);
    dim3 blkDim(size / 2, 1, 1);

    JudgeSwap<int> is_swap;

    bubbleSort<int> <<<grdDim, blkDim>>>(d_v, size, is_swap);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemcpy(h_v, d_v, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_v);

    for (int i = 0; i < size; i++) {
        std::cout << (i == 0 ? "{" : "") << h_v[i] << (i < size - 1 ? " ," : "}");
    }
    std::cout << std::endl;

    return 0;
}

