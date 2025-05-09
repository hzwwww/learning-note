#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int nElem = 1024;

    // 定义1024线程的block
    dim3 block (1024);
    // 定义n个block的grid
    // 向上取整：1025条数据，需要2个block，1024条数据，需要1个block
    dim3 grid ((nElem+block.x-1)/block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    block.x = 512;
    grid.x = (nElem+block.x-1)/block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    block.x = 256;
    grid.x = (nElem+block.x-1)/block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    block.x = 128;
    grid.x = (nElem+block.x-1)/block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    mcDeviceReset();
    return(0);
}