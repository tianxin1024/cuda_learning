# CUDA相关文档记录

## 1. CUDA常见优化策略
    1. 异步执行与流(streams)
    2. 共享内存(shared memory)
    3. 内存对齐(memory alignment)
    4. 循环展开(Loop Unrolling)
    5. 使用更快的数学函数
        (sin, cos, exp) -> (_sinf, _cosf, _expf)
    6. 优化内存访问模式
        * 确保相邻线程访问的内存是连续的，以利用合并内存访问的优势
        * Page-Locked (Pinned) Memory: 使用页锁定内存（cudaMallocHost）可以显著加快主机与设备间的内存传输速度
    7. 避免线程分歧(Thread Divergence) (bank 冲突)
        尽量确保warp内的线程执行相同的代码路径，以避免线程分歧导致的性能下降
    8. 使用适当的CUDA编译优化选项
        在编译CUDA代码时，可以使用如-O3、--ptxas-options=-v等编译器优化选项来提高代码性能。
    9. 减少全局内存访问
        如果可能，尽量减少全局内存的访问次数。例如，可以将经常访问的数据存储在局部变量或寄存器中。
    10. 利用常量内存(Constant Memory)
        如果某些数据在kernel执行过程中不会改变，可以将其存储在常量内存中，以提高访问速度。
    11. 指令缓存(Instruction Cache)
        对于计算能力较高的GPU，可以利用指令缓存来提高指令获取的速度。
    12. 分析性能瓶颈
        使用NVIDIA的Nsight工具或其他CUDA性能分析工具来识别程序中的瓶颈，并针对这些瓶颈进行优化。
    13. 算法层面的优化

