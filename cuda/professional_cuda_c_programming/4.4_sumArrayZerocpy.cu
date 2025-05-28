#include "common.h"
#include <__clang_maca_builtin_vars.h>
#include <cuda_runtime.h>
#include <stdio.h>

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                    gpuRef[i], i);
            break;
        }
    }

    return;
}

void initialData(float *ip, int size)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArrays(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}

__global__ void sumArraysZeroCopy(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv)
{
    int dev = 0;
    CHECK(mcSetDevice(dev));

    mcDeviceProp_t deviceProp;
    CHECK(mcGetDeviceProperties(&deviceProp, dev));

    if (!deviceProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(mcDeviceReset());
        exit(EXIT_SUCCESS);
    }

    printf("Using Device %d: %s ", dev, deviceProp.name);

    int ipower = 10;

    if (argc > 1) ipower = atoi(argv[1]);

    int nElem = 1 << ipower;
    size_t nBytes = nElem * sizeof(float);

    if (ipower < 18)
    {
        printf("Vector size %d power %d  nbytes  %3.0f KB\n", nElem, ipower,
               (float)nBytes / (1024.0f));
    }
    else
    {
        printf("Vector size %d power %d  nbytes  %3.0f MB\n", nElem, ipower,
               (float)nBytes / (1024.0f * 1024.0f));
    }

    // part 1: 使用设备内存
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    float *d_A, *d_B, *d_C;
    CHECK(mcMalloc((float**)&d_A, nBytes));
    CHECK(mcMalloc((float**)&d_B, nBytes));
    CHECK(mcMalloc((float**)&d_C, nBytes));

    CHECK(mcMemcpy(d_A, h_A, nBytes, mcMemcpyHostToDevice));
    CHECK(mcMemcpy(d_B, h_B, nBytes, mcMemcpyHostToDevice));

    int iLen = 512;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x);

    int start = seconds();
    sumArrays<<<grid, block>>>(d_A, d_B, d_C, nElem);
    mcDeviceSynchronize();
    printf("sumArrays took %f secs\n", seconds()-start);

    CHECK(mcMemcpy(gpuRef, d_C, nBytes, mcMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, nElem);

    CHECK(mcFree(d_A));
    CHECK(mcFree(d_B));

    free(h_A);
    free(h_B);

    // part 2：使用零拷贝内存
    // 申请零拷贝内存
    CHECK(mcMallocManaged((void **)&h_A, nBytes));
    CHECK(mcMallocManaged((void **)&h_B, nBytes));

    initialData(h_A, nElem);
    initialData(h_B, nElem);
    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // 获取设备端指向零拷贝内存的指针
    CHECK(mcHostGetDevicePointer((void **)&d_A, (void *)h_A, 0));
    CHECK(mcHostGetDevicePointer((void **)&d_B, (void *)h_B, 0));

    start = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    printf("sumArraysOnHost took %f secs\n", seconds()-start);

    start = seconds();
    sumArraysZeroCopy<<<grid, block>>>(d_A, d_B, d_C, nElem);
    mcDeviceSynchronize();
    printf("sumArraysZeroCopy took %f secs\n", seconds()-start);

    CHECK(mcMemcpy(gpuRef, d_C, nBytes, mcMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, nElem);

    // free  memory
    CHECK(mcFree(d_C));
    CHECK(mcFree(h_A));
    CHECK(mcFree(h_B));

    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(mcDeviceReset());
    return EXIT_SUCCESS;
}