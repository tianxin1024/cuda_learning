ncu 分析策略：


## kernel 1
```
ncu --kernel-name softmax_forward_kernel1 --launch-skip 4 --launch-count 1 ./softmax 1
# --kernel-name : 分析的kernel函数
# --launch-skip : 跳过前 N 次 kernel 启动 
# --launch-count: 只分析前 N 次 kernel 启动
```

### 分析结果：
```
block_size   32 | time 4.7273 ms | per token 0.58 µs
block_size   64 | time 4.6205 ms | per token 0.56 µs
block_size  128 | time 5.0759 ms | per token 0.62 µs
block_size  256 | time 5.7979 ms | per token 0.71 µs
block_size  512 | time 7.8459 ms | per token 0.96 µs
block_size 1024 | time 16.7985 ms | per token 2.05 µs
==PROF== Disconnected from process 543294
[543294] softmax@127.0.0.1
  softmax_forward_kernel1(float *, const float *, int, int) (16, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         6.98
    SM Frequency            cycle/usecond       913.58
    Elapsed Cycles                  cycle   13,843,531
    Memory Throughput                   %        38.14
    DRAM Throughput                     %        10.50
    Duration                      msecond        15.15
    L1/TEX Cache Throughput             %        77.09
    L2 Cache Throughput                 %        38.14
    SM Active Cycles                cycle 4,788,773.59
    Compute (SM) Throughput             %         5.27
    ----------------------- ------------- ------------

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   512
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                     16
    Registers Per Thread             register/thread              39
    Shared Memory Configuration Size           Kbyte            8.19
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    Threads                                   thread           8,192
    Waves Per SM                                                0.12
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            3
    Block Limit Shared Mem                block            8
    Block Limit Warps                     block            3
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        33.30
    Achieved Active Warps Per SM           warp        15.99
    ------------------------------- ----------- ------------

```

## kernel 2
```
ncu --kernel-name softmax_forward_kernel2 --launch-skip 4 --launch-count 1 ./softmax 2
# --kernel-name : 分析的kernel函数
# --launch-skip : 跳过前 N 次 kernel 启动 
# --launch-count: 只分析前 N 次 kernel 启动
```

```
block_size   32 | time 2.1972 ms | per token 0.27 µs
block_size   64 | time 1.9764 ms | per token 0.24 µs
block_size  128 | time 1.9651 ms | per token 0.24 µs
block_size  256 | time 1.4582 ms | per token 0.18 µs
block_size  512 | time 0.8298 ms | per token 0.10 µs
block_size 1024 | time 1.3016 ms | per token 0.16 µs
==PROF== Disconnected from process 543833
[543833] softmax@127.0.0.1
  softmax_forward_kernel2(float *, const float *, int, int) (8192, 1, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         6.99
    SM Frequency            cycle/usecond       915.05
    Elapsed Cycles                  cycle      865,601
    Memory Throughput                   %        70.45
    DRAM Throughput                     %        69.81
    Duration                      usecond       945.95
    L1/TEX Cache Throughput             %        70.93
    L2 Cache Throughput                 %        62.03
    SM Active Cycles                cycle   859,638.35
    Compute (SM) Throughput             %        70.45
    ----------------------- ------------- ------------

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                   512
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                  8,192
    Registers Per Thread             register/thread              22
    Shared Memory Configuration Size           Kbyte           16.38
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block      Kbyte/block            2.05
    Static Shared Memory Per Block        byte/block               0
    Threads                                   thread       4,194,304
    Waves Per SM                                               59.36
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            5
    Block Limit Shared Mem                block            5
    Block Limit Warps                     block            3
    Theoretical Active Warps per SM        warp           48
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        97.01
    Achieved Active Warps Per SM           warp        46.56
    ------------------------------- ----------- ------------
```

## kernel 3
```
ncu --kernel-name softmax_forward_kernel3 --launch-skip 4 --launch-count 1 ./softmax 3
# --kernel-name : 分析的kernel函数
# --launch-skip : 跳过前 N 次 kernel 启动 
# --launch-count: 只分析前 N 次 kernel 启动
```

```
block_size   32 | time 2.6630 ms | per token 0.33 µs
block_size   64 | time 2.5669 ms | per token 0.31 µs
block_size  128 | time 2.5652 ms | per token 0.31 µs
block_size  256 | time 2.5703 ms | per token 0.31 µs
block_size  512 | time 2.5840 ms | per token 0.32 µs
block_size 1024 | time 2.5723 ms | per token 0.31 µs
==PROF== Disconnected from process 544414
[544414] softmax@127.0.0.1
  softmax_forward_kernel3(float *, const float *, int, int) (8192, 1, 1)x(32, 1, 1), Context 1, Stream 7, Device 0, CC 8.6
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         7.00
    SM Frequency            cycle/usecond       915.27
    Elapsed Cycles                  cycle    3,004,364
    Memory Throughput                   %        55.98
    DRAM Throughput                     %        55.98
    Duration                      msecond         3.28
    L1/TEX Cache Throughput             %        18.21
    L2 Cache Throughput                 %        28.14
    SM Active Cycles                cycle 2,913,056.91
    Compute (SM) Throughput             %        12.02
    ----------------------- ------------- ------------

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                    32
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                  8,192
    Registers Per Thread             register/thread              18
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block             128
    Static Shared Memory Per Block        byte/block               0
    Threads                                   thread         262,144
    Waves Per SM                                               11.13
    -------------------------------- --------------- ---------------

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block           84
    Block Limit Shared Mem                block           28
    Block Limit Warps                     block           48
    Theoretical Active Warps per SM        warp           16
    Theoretical Occupancy                     %        33.33
    Achieved Occupancy                        %        33.02
    Achieved Active Warps Per SM           warp        15.85
    ------------------------------- ----------- ------------
```


