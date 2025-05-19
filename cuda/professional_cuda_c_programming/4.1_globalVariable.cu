#include <cuda_runtime.h>
#include <stdio.h>

__device__ float devData;

__global__ void checkGlobalVariable() 
{
    printf("Device: the value of the global variable is %f\n", devData);
    devData += 2.0f;
}

int main(void) {
    float value = 3.14f;
    // 将数据从主机拷贝到设备端
    // 与cudaMemcpy的区别是，cudaMemcpy返回的是主机端的一个指针，指向了device的内存
    // mcMemcpyToSymbol操作的是__device__, __constant__
    // host端无法访问device端的变量，需要拷贝到host后，才能访问，反之亦然
    mcMemcpyToSymbol(devData, &value, sizeof(float)); 
    printf("Host: copied %f to the global variable\n", value);

    checkGlobalVariable<<<1, 1>>>();

    mcMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("Host: the value changed by the kernel to %f\n", value);

    mcDeviceReset();
    return EXIT_SUCCESS;
}