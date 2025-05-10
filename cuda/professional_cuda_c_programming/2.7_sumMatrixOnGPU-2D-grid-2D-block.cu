#include <cuda_runtime.h>
#include <stdio.h>
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

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy=0; iy<ny; iy++) {
        for (int ix=0; ix<nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, const int nx, const int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
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

    // 设置矩阵大小
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    printf("Matrix size (%d, %d)\n", nx, ny);

    // 申请host内存
    size_t nBytes = ny * nx * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes); // 保存host的计算结果
    gpuRef = (float *)malloc(nBytes); // 保存device的计算结果（从device拷贝）

    // 初始化数据
    initialData(h_A, nxy);
    initialData(h_B, nxy);

    // 数据置0，保证状态正确，防止出现脏数据，便于调试（0代表数据未填充）
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // 申请device全局内存
    float *d_MatA, *d_MatB, *d_MatC;
    mcMalloc(&d_MatA, nBytes);
    mcMalloc(&d_MatB, nBytes);
    mcMalloc(&d_MatC, nBytes);

    // 复制host数据到device
    mcMemcpy(d_MatA, h_A, nBytes, mcMemcpyHostToDevice);
    mcMemcpy(d_MatB, h_B, nBytes, mcMemcpyHostToDevice);

    // 在host端调用kernel计算向量和
    int dimx = 32;
    int dimy = 16;
    dim3 block (dimx, dimy);
    dim3 grid ((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    // 对kernel调用计时
    double iStart, iElaps;
    iStart = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    mcDeviceSynchronize();
    iElaps = cpuSecond() - iStart;

    printf("Execution configuration <<<%d %d, %d, %d>>>, Time elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    // 复制device数据到host
    // kernel是异步调用的，但是mcMemcpy会感知数据依赖，等待相关kernel执行完成后，再执行拷贝，保证数据已全部就绪
    mcMemcpy(gpuRef, d_MatC, nBytes, mcMemcpyDeviceToHost);

    // 在host端计算向量和
    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;

    printf("Execution configuration CPU, Time elapsed %f sec\n", iElaps);

    // 检查device计算结果
    checkResult(hostRef, gpuRef, nxy);

    // 释放device全局内存
    mcFree(d_MatA);
    mcFree(d_MatB);
    mcFree(d_MatC);

    // 释放host内存
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    mcDeviceReset();

    return(0);
}