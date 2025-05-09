#include <__clang_maca_builtin_vars.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void) {
    printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, %d) gridDim: (%d, %d, %d)\n", 
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z,
        blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z
    );
}

int main() {
    int nElem = 6;

    // 包含3个线程的一维block
    dim3 block (3);
    dim3 grid ((nElem+block.x-1)/block.x);

    // 在host侧检查dimesion
    printf("grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("block: (%d, %d, %d)\n", block.x, block.y, block.z);

    // 在device侧检查dimesion
    checkIndex<<<grid, block>>>();

    // 清理该设备上的所有运行时状态，包括正在运行的kernel、申请的显存、错误状态、运行状态
    mcDeviceReset();

    return 0;
}