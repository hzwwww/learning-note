#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                         \
{                                                                           \
    const mcError_t error = call;                                           \
    if (error != mcSuccess) {                                               \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code: %d, reason: %s\n", error, mcGetErrorString(error));   \
        exit(1);                                                            \
    }                                                                       \
}                                                                           \

void initialInt(int *ip, int size) {
    for (int i=0; i<size; i++) {
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny) {
    int *ic = C;
    printf("\nMatrix: (%d, %d)\n", nx, ny);
    for (int iy=0; iy<ny; iy++) {
        for (int ix=0; ix<nx; ix++) {
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf("thread_id (%d, %d) block_id (%d, %d) coordinate (%d, %d) "
        "global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x,
        blockIdx.y, ix, iy, idx, A[idx]);
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    int dev = 0;
    mcDeviceProp_t deviceProp;
    CHECK(mcGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(mcSetDevice(dev));

    // 设置矩阵维度
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    // 申请host内存
    int *h_A;
    h_A = (int *)malloc(nBytes);

    // 初始化矩阵
    initialInt(h_A, nxy);
    
    // 打印矩阵
    printMatrix(h_A, nx, ny);

    // 申请device全局内存
    // 矩阵在device中也是线性存储
    int *d_MatA;
    mcMalloc((void **)&d_MatA, nBytes);

    // 拷贝数据到device
    mcMemcpy(d_MatA, h_A, nBytes, mcMemcpyHostToDevice);

    // 设置执行配置
    dim3 block(4, 2);
    // 由于一个线程处理一条数据，所以需要根据block配置调整grid配置，让线程矩阵足够大
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    // 调用kernel
    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
    mcDeviceSynchronize();

    // 释放host和device内存
    free(h_A);
    mcFree(d_MatA);

    mcDeviceReset();

    return(0);
}
