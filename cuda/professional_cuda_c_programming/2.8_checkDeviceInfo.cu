#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);

    // 获取device数量
    int deviceCount = 0;
    mcError_t error_id = mcGetDeviceCount(&deviceCount);
    if (error_id != mcSuccess) {
        printf("mcGetDeviceCount returned %d\n -> %s\n", 
            (int)error_id, mcGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    // 获取device属性
    dev = 0;
    mcSetDevice(dev);
    mcDeviceProp_t deviceProp;
    mcGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);

    // CUDA driver, runtime版本
    mcDriverGetVersion(&driverVersion);
    mcRuntimeGetVersion(&runtimeVersion);
    printf("    CUDA Driver Version / Runtime Version   %d.%d / %d.%d\n",
        driverVersion/1000, (driverVersion%100)/10,
        runtimeVersion/1000, (runtimeVersion%100)/10);

    // CUDA Capalility major, minor版本
    // 主版本号（Major）：代表 GPU 的核心架构（如 7.x 对应 Volta，8.x 对应 Ampere）。
    // 次版本号（Minor）：同一架构下的增量改进（如新增指令或优化）。
    printf("    CUDA Capalility Major / Minor version number   %d.%d\n",
        deviceProp.major, deviceProp.minor);

    // 全局内存，即GPU显存
    printf("    Total amount of global memory:  %.2f GBytes (%llu bytes)\n",
        (float) deviceProp.totalGlobalMem/(pow(1024.0, 3)),
        (unsigned long long) deviceProp.totalGlobalMem);

    // GPU时钟
    printf("    GPU Clock rate: %.0f MHz (%0.2f GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    
    // 显存时钟
    printf("    Memory Clock rate: %.0f MHz\n",
        deviceProp.memoryClockRate * 1e-3f);

    // 显存总线长度
    printf("    Memory Bus Width: %d bits\n",
        deviceProp.memoryBusWidth);

    // L2缓存
    // L1缓存：每个SM内部，容量最小，速度最快，用于缓存线程块（Block）的局部数据
    // L2缓存：全局，位于GPU core和vram之间，容量较大，速度较快，用于缓存全局数据，减少显存访问
    // vram：独立于 GPU core，容量最大，速度最慢，用于存储所有 GPU 数据
    if (deviceProp.l2CacheSize) {
        printf("    L2 Cache Size: %d bytes\n",
            deviceProp.l2CacheSize);
    }

    // 最大纹理维度
    // maxTexture3D为0，表示GPU不支持3D渲染
    printf("    Max Texture Dimension Size (x,y,z) 1D=(%d), 2D=(%d,%d), 3D(%d,%d,%d)\n",
        deviceProp.maxTexture1D, 
        deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
        deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);

    // 最大一维分层纹理尺寸
    printf("    Max Layered Texture Size (dim) x layers 1D=(%d) x %d, 2D=(%d,%d) x %d\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[0],
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

    // 常量内存
    // 用于存储少量只读数据，具有读取速度快的特点
    printf("    Total amount of constant memory: %lu bytes\n", 
        deviceProp.totalConstMem);
    
    // block共享内存
    // 位于SM内部，访问速度比全局内存快 10~100 倍
    printf("    Total amount of shared memory per block: %lu bytes\n", 
        deviceProp.sharedMemPerBlock);

    // block寄存器数量
    printf("    Total number of registers available per block: %d\n", 
        deviceProp.regsPerBlock);

    // Warp size
    // GPU调度和执行线程的最小单位，包含固定数量的线程，由SM统一管理
    printf("    Warp size: %d\n", deviceProp.warpSize);

    // SM最大线程数
    printf("    Max number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    
    // block最大线程数
    printf("    Max number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);

    // block每个维度最大线程数
    printf("    Max size of each dimension of a block: %d x %d x %d\n",
        deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

    // grid每个维度最大block数
    printf("    Max size of each dimension of a grid: %d x %d x %d\n",
        deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
          
    // 内存间距
    // 用于对齐内存，合并内存访问，提高访问速度
    printf("    Max memory pitch: %lu bytes\n", deviceProp.memPitch);

    exit(EXIT_SUCCESS);
}