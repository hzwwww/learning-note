#include "./common.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void mathKernel1(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if (tid % 2 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel2(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0) {
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel3(float *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);

    if (ipred) {
        ia = 100.0f;
    }

    if (!ipred) {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel4(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;

    if (itid & 0x01 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void warmingup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

int main(int argc, char **argv) {
    int dev = 0;
    mcDeviceProp_t deviceProp;
    mcGetDeviceProperties(&deviceProp, dev);
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // 设置数据、block大小
    int size = 64;
    int blocksize = 64;
    if (argc > 1) blocksize = atoi(argv[1]);
    if (argc > 2) size = atoi(argv[2]);
    printf("Data size %d ", size);

    // 设置执行配置
    dim3 block (blocksize, 1);
    dim3 grid ((size+block.x-1)/block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // 申请device内存
    float *d_C;
    size_t nBytes = size * sizeof(float);
    mcMalloc((float**)&d_C, nBytes);

    // 预热
    size_t iStart, iElaps;
    mcDeviceSynchronize();
    iStart = seconds();
    warmingup<<<grid, block>>> (d_C);
    mcDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("warmup <<<%4d, %4d>>> elapsed %zul sec \n", grid.x, block.x, iElaps);

    // 运行kernel 1
    iStart = seconds();
    mathKernel1<<<grid, block>>> (d_C);
    mcDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel1 <<<%4d, %4d>>> elapsed %zul sec \n", grid.x, block.x, iElaps);

    // 运行kernel 2
    iStart = seconds();
    mathKernel2<<<grid, block>>> (d_C);
    mcDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel2 <<<%4d, %4d>>> elapsed %zul sec \n", grid.x, block.x, iElaps);

    // 运行kernel 3
    iStart = seconds();
    mathKernel3<<<grid, block>>> (d_C);
    mcDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel3 <<<%4d, %4d>>> elapsed %zul sec \n", grid.x, block.x, iElaps);

    // 运行kernel 4
    iStart = seconds();
    mathKernel4<<<grid, block>>> (d_C);
    mcDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel4 <<<%4d, %4d>>> elapsed %zul ns \n", grid.x, block.x, iElaps);

    // 释放资源
    mcFree(d_C);
    mcDeviceReset();

    return EXIT_SUCCESS;
}