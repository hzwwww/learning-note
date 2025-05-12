#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
    int iDev = 0;
    mcDeviceProp_t iProp;
    mcGetDeviceProperties(&iProp, iDev);

    printf("Device %d: %s\n", iDev, iProp.name);
    printf("Number of multiprocessors: %d\n", iProp.multiProcessorCount);
    printf("Total amount of constant memory: %4.2f KB\n", iProp.totalConstMem/1024.0);
    printf("Total amount of registers available per block: %d\n", iProp.regsPerBlock);
    printf("Warp size %d\n", iProp.warpSize);
    printf("Max number of threads per block: %d\n", iProp.maxThreadsPerBlock);
    printf("Max number of threads per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor);
    printf("Max number of warps per multiprocessor: %d\n", iProp.maxThreadsPerMultiProcessor / iProp.warpSize);

    return EXIT_SUCCESS;
}