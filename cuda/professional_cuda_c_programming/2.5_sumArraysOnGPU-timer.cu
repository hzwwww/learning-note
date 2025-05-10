#include <__clang_maca_builtin_vars.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>

#define CHECK(call)                                                         \
{                                                                           \
    const mcError_t error = call;                                           \
    if (error != mcSuccess) {                                               \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code: %d, reason: %s\n", error, mcGetErrorString(error));   \
        exit(1);                                                            \
    }                                                                       \
}                                                                           \

void checkResult(float *hostRef, float *gpuRef, const int N) {
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i=0; i<N; i++) {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match. \n\n");
}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned) time(&t));

    for (int i=0; i<size; i++) {
        ip[i] = (float) (rand() & 0xFF)/10.0f;
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
    for (int i=0; i<N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
    // 使用线程id建立线程与数据的映射
    // 线程id计算方式：第几个block * 每个block的线程数 + 在当前block的线程id
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 处理线程数 > 矢量大小的情况
    if (i < N) C[i] = A[i] + B[i];
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    int dev = 0;

    // 打印device状态
    mcDeviceProp_t deviceProp;
    mcGetDeviceProperties(&deviceProp, dev);
    printf("Using device %s, maxThreadsPerBlock %d\n", deviceProp.name, deviceProp.maxThreadsPerBlock);

    // 绑定device
    mcSetDevice(dev);

    // 设置向量大小
    int nElem = 1<<24;
    printf("Vector size %d\n", nElem);

    // 申请host内存
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes); // 保存host的计算结果
    gpuRef = (float *)malloc(nBytes); // 保存device的计算结果（从device拷贝）

    // 初始化数据
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    // 数据置0，保证状态正确，防止出现脏数据，便于调试（0代表数据未填充）
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // 申请device全局内存
    float *d_A, *d_B, *d_C;
    mcMalloc(&d_A, nBytes);
    mcMalloc(&d_B, nBytes);
    mcMalloc(&d_C, nBytes);

    // 复制host数据到device
    mcMemcpy(d_A, h_A, nBytes, mcMemcpyHostToDevice);
    mcMemcpy(d_B, h_B, nBytes, mcMemcpyHostToDevice);

    // 在host端调用kernel计算向量和
    int iLen = 512;
    dim3 block (iLen); // 单维block，每个block处理1024个元素
    dim3 grid ((nElem+iLen-1)/block.x); // 根据每个block处理的元素数量，计算grid所需的block数

    // 对kernel调用计时
    double iStart, iElaps;
    iStart = cpuSecond();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    mcDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    printf("Execution configuration <<<%d, %d>>>, Time elapsed %f sec\n", grid.x, block.x, iElaps);

    // 复制device数据到host
    // kernel是异步调用的，但是mcMemcpy会感知数据依赖，等待相关kernel执行完成后，再执行拷贝，保证数据已全部就绪
    mcMemcpy(gpuRef, d_C, nBytes, mcMemcpyDeviceToHost);

    // 在host端计算向量和
    iStart = cpuSecond();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = cpuSecond() - iStart;

    printf("Execution configuration CPU, Time elapsed %f sec\n", iElaps);

    // 检查device计算结果
    checkResult(hostRef, gpuRef, nElem);

    // 释放device全局内存
    mcFree(d_A);
    mcFree(d_B);
    mcFree(d_C);

    // 释放host内存
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return(0);
}