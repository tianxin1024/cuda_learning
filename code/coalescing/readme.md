# 高效访问全局内存

### 说明
研究如何将一组数据线程访问的全局内存合并到一个事务中，以及对齐和跨步如何影响CUDA各个硬件的合并。
对于最新版本的 CUDA 硬件，未对齐的数据访问不是一个大问题。然而，不管 CUDA 硬件是如何产生的，在全局内存中大步前进都是有问题的，而且在许多情况下似乎是不可避免的，

![link](https://developer.nvidia.com/zh-cn/blog/how-access-global-memory-efficiently-cuda-c-kernels/)

