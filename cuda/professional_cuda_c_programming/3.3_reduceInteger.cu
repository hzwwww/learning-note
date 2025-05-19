#include "./common.h"
#include <__clang_maca_builtin_vars.h>
#include <__clang_maca_device_functions.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

// 交错配对，递归
int recursiveReduce(int *data, int const size)
{
    if (size == 1) return data[0];

    int const stride = size / 2;

    for (int i=0; i < stride; i++) {
        data[i] += data[i+stride];
    }

    return recursiveReduce(data, stride);
}

// 相邻配对，并行，分化严重
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算本block对应数据的指针
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx > n) return;

    for (int stride = 1; stride < blockDim.x; stride *=2) 
    {
        // 对于每个Warp的线程，都有一半的线程处于工作，一半处于空闲状态，分化严重
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 相邻配对，并行，分化改善
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx > n) return;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;

        // 把计算集中在前n个Warp中，保证对于每个Warp的线程，要么全部工作，要么全部闲置，改善了Warp的分化
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 交错配对，并行，分化改善，内存访问连续
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n) 
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx > n) return;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) 
        {
            // 内存访问时连续的，可大大提升性能
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 展开，一个block处理2个数据块
__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n) 
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) 
        {
            // 内存访问时连续的，可大大提升性能
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 展开，一个block处理4个数据块
__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n) 
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    if (idx + blockDim.x * 3 < n) 
    {
        idata[tid] += idata[tid + blockDim.x];
        idata[tid] += idata[tid + blockDim.x * 2];
        idata[tid] += idata[tid + blockDim.x * 3];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) 
        {
            // 内存访问时连续的，可大大提升性能
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 展开，一个block处理8个数据块
__global__ void reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n) 
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x * 7 < n) 
    {
        idata[tid] += idata[tid + blockDim.x];
        idata[tid] += idata[tid + blockDim.x * 2];
        idata[tid] += idata[tid + blockDim.x * 3];
        idata[tid] += idata[tid + blockDim.x * 4];
        idata[tid] += idata[tid + blockDim.x * 5];
        idata[tid] += idata[tid + blockDim.x * 6];
        idata[tid] += idata[tid + blockDim.x * 7];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) 
        {
            // 内存访问时连续的，可大大提升性能
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 展开最后32个线程
// |---------------------------array--------------------------|
// |-------block------||-------block------||-------block------|
// |-warp-|....|-warp-|
// |tid<32|
__global__ void reduceUnrollWarp8(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x * 7 < n) 
    {
        idata[tid] += idata[tid + blockDim.x];
        idata[tid] += idata[tid + blockDim.x * 2];
        idata[tid] += idata[tid + blockDim.x * 3];
        idata[tid] += idata[tid + blockDim.x * 4];
        idata[tid] += idata[tid + blockDim.x * 5];
        idata[tid] += idata[tid + blockDim.x * 6];
        idata[tid] += idata[tid + blockDim.x * 7];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride) 
        {
            // 内存访问时连续的，可大大提升性能
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // 当剩余64个元素时，用一个warp处理
    // 利用warp内线程的隐式同步特性，可以不用__syncthreads()
    // 64个元素求和，仅需63步计算，这里共32*6步计算，有很多无效计算，但不影响最终结果
    if (tid < 32) 
    {
        // volatile变量
        // 无读写优化：不会存储在缓存或寄存器中
        // 多线程安全：读写标志位对多线程可见
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32]; // 将后32个元素一对一加到前32个元素
        vmem[tid] += vmem[tid + 16]; // 将后16个元素一对一加到前16个元素
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1]; // 将后1个元素一对一加到前1个元素
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// 展开block所有的线程
__global__ void reduceCompleteUnrollWarp8(int *g_idata, int *g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    if (idx + blockDim.x * 7 < n) 
    {
        idata[tid] += idata[tid + blockDim.x];
        idata[tid] += idata[tid + blockDim.x * 2];
        idata[tid] += idata[tid + blockDim.x * 3];
        idata[tid] += idata[tid + blockDim.x * 4];
        idata[tid] += idata[tid + blockDim.x * 5];
        idata[tid] += idata[tid + blockDim.x * 6];
        idata[tid] += idata[tid + blockDim.x * 7];
    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid+512];
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid+256];
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid+128];
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid+64];
    __syncthreads();

    // 这里也可以一直if (blockDim.x >= xx && tid < xx)，直到最后两个线程，但是warp分化太严重
    if (tid < 32) 
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata,
                                     unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  idata[tid] += idata[tid + 256];

    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  idata[tid] += idata[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


int main(int argc, char **argv)
{
    // 绑定device
    int dev = 0;
    mcDeviceProp_t deviceProp;
    CHECK(mcGetDeviceProperties(&deviceProp, dev));
    printf("%s Starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(mcSetDevice(dev));

    bool bResult = false;

    int size = 1 << 24;
    printf("    with array size %d ", size);

    // 运行配置
    int blocksize = 512;

    if (argc > 1) {
        blocksize = atoi(argv[1]);
    }

    dim3 block (blocksize, 1);
    dim3 grid ((size + block.x - 1)/block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // 申请host内存
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp = (int *) malloc(bytes);

    // 初始化数据
    for (int i=0; i < size; i++) {
        h_idata[i] = (int)(rand() & 0xFF);
    }

    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // 申请device内存
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(mcMalloc((void **) &d_idata, bytes));
    CHECK(mcMalloc((void **) &d_odata, grid.x * sizeof(int)));

    // cpu reduce
    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce  elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1: reduceNeighbored
    CHECK(mcMemcpy(d_idata, h_idata, bytes, mcMemcpyHostToDevice));
    CHECK(mcDeviceSynchronize());
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(mcDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(mcMemcpy(h_odata, d_odata, grid.x * sizeof(int), mcMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i=0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", 
        iElaps, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighbored with less divergence
    CHECK(mcMemcpy(d_idata, h_idata, bytes, mcMemcpyHostToDevice));
    CHECK(mcDeviceSynchronize());
    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(mcDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(mcMemcpy(h_odata, d_odata, grid.x * sizeof(int), mcMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i=0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", 
        iElaps, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    CHECK(mcMemcpy(d_idata, h_idata, bytes, mcMemcpyHostToDevice));
    CHECK(mcDeviceSynchronize());
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(mcDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(mcMemcpy(h_odata, d_odata, grid.x * sizeof(int), mcMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i=0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu reduceInterleaved elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", 
        iElaps, gpu_sum, grid.x, block.x);

    // kernel 4: reduceUnrolling2
    CHECK(mcMemcpy(d_idata, h_idata, bytes, mcMemcpyHostToDevice));
    CHECK(mcDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling2<<<grid.x/2, block>>>(d_idata, d_odata, size);
    CHECK(mcDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(mcMemcpy(h_odata, d_odata, grid.x/2 * sizeof(int), mcMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i=0; i < grid.x/2; i++) gpu_sum += h_odata[i];

    printf("gpu reduceUnrolling2 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", 
        iElaps, gpu_sum, grid.x, block.x);

    // kernel 4: reduceUnrolling4
    CHECK(mcMemcpy(d_idata, h_idata, bytes, mcMemcpyHostToDevice));
    CHECK(mcDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling4<<<grid.x/4, block>>>(d_idata, d_odata, size);
    CHECK(mcDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(mcMemcpy(h_odata, d_odata, grid.x/4 * sizeof(int), mcMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i=0; i < grid.x/4; i++) gpu_sum += h_odata[i];

    printf("gpu reduceUnrolling4 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", 
        iElaps, gpu_sum, grid.x, block.x);

    // kernel 5: reduceUnrolling8
    CHECK(mcMemcpy(d_idata, h_idata, bytes, mcMemcpyHostToDevice));
    CHECK(mcDeviceSynchronize());
    iStart = seconds();
    reduceUnrolling8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    CHECK(mcDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(mcMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), mcMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i=0; i < grid.x/8; i++) gpu_sum += h_odata[i];

    printf("gpu reduceUnrolling8 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", 
        iElaps, gpu_sum, grid.x, block.x);

    // kernel 6: reduceUnrollWarp8
    CHECK(mcMemcpy(d_idata, h_idata, bytes, mcMemcpyHostToDevice));
    CHECK(mcDeviceSynchronize());
    iStart = seconds();
    reduceUnrollWarp8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    CHECK(mcDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(mcMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), mcMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i=0; i < grid.x/8; i++) gpu_sum += h_odata[i];

    printf("gpu reduceUnrollWarp8 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", 
        iElaps, gpu_sum, grid.x, block.x);

    // kernel 7: reduceCompleteUnrollWarp8
    CHECK(mcMemcpy(d_idata, h_idata, bytes, mcMemcpyHostToDevice));
    CHECK(mcDeviceSynchronize());
    iStart = seconds();
    reduceCompleteUnrollWarp8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    CHECK(mcDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(mcMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int), mcMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i=0; i < grid.x/8; i++) gpu_sum += h_odata[i];

    printf("gpu reduceCompleteUnrollWarp8 elapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n", 
        iElaps, gpu_sum, grid.x, block.x);

    // kernel 9: reduceCompleteUnroll
    CHECK(mcMemcpy(d_idata, h_idata, bytes, mcMemcpyHostToDevice));
    CHECK(mcDeviceSynchronize());
    iStart = seconds();

    // device函数的分支消耗远比host函数大，所以尽量将if、switch分支放在host端
    switch (blocksize)
    {
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 512:
        reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 256:
        reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 128:
        reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 64:
        reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    }

    CHECK(mcDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(mcMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     mcMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll   elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);


    // 释放内存
    free(h_idata);
    free(h_odata);
    free(tmp);

    mcFree(d_idata);
    mcFree(d_odata);

    mcDeviceReset();
    
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}