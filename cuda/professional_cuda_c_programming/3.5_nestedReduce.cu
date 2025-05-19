#include "./common.h"
#include <__clang_maca_builtin_vars.h>
#include <__clang_maca_device_functions.h>
#include <stdio.h>
#include <cuda_runtime.h>

int cpuRecursiveReduce(int *data, int const size)
{
    if (size == 1) return data[0];

    int const stride = size / 2;

    for (int i=0; i < stride; i++) {
        data[i] += data[i+stride];
    }

    return cpuRecursiveReduce(data, stride);
}

__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize) 
{
    unsigned int tid = threadIdx.x;

    int *idata = g_idata + blockDim.x * blockIdx.x;
    int *odata = &g_odata[blockIdx.x];

    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int istride = isize >> 1;
    if (istride > 1 && tid < istride) {
        idata[tid] += idata[tid + istride];
    }

    // 同步block下的所有线程
    __syncthreads();

    if (tid == 0) {
        gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
        mcDeviceSynchronize();
    }

    // 再次同步block下的所有线程
    __syncthreads();
}

// 由于子线程和父线程看到的数据是一样的，所以启动子网络前，无需等待其他线程执行完
__global__ void gpuRecursiveReduceNosync(int *g_idata, int *g_odata, unsigned int isize) 
{
    unsigned int tid = threadIdx.x;

    int *idata = g_idata + blockDim.x * blockIdx.x;
    int *odata = &g_odata[blockIdx.x];

    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int istride = isize >> 1;
    if (istride > 1 && tid < istride) {
        idata[tid] += idata[tid + istride];
        if (tid == 0) {
            gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
        }
    }
}

// 尝试一个子grid处理所有数据，而不是部分数据，减少子grid创建数量
__global__ void gpuRecursiveReduce2(int *g_idata, int *g_odata, int iStride, int const iDim) 
{
    int *idata = g_idata + blockIdx.x * iDim;

    if (iStride == 1 && threadIdx.x == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    idata[threadIdx.x] += idata[threadIdx.x + iStride];

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        gpuRecursiveReduce2<<<gridDim.x, iStride/2>>>(g_idata, g_odata, iStride/2, iDim);
    }
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0, gpu_sum;
    mcDeviceProp_t deviceProp;
    CHECK(mcGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(mcSetDevice(dev));

    bool bResult = false;

    // set up execution configuration
    int nblock  = 2048;
    int nthread = 512;   // initial block size

    if(argc > 1)
    {
        nblock = atoi(argv[1]);   // block size from command line argument
    }

    if(argc > 2)
    {
        nthread = atoi(argv[2]);   // block size from command line argument
    }

    int size = nblock * nthread; // total number of elements to reduceNeighbored

    dim3 block (nthread, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("array %d grid %d block %d\n", size, grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)( rand() & 0xFF );
        h_idata[i] = 1;
    }

    memcpy(tmp, h_idata, bytes);

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(mcMalloc((void **) &d_idata, bytes));
    CHECK(mcMalloc((void **) &d_odata, grid.x * sizeof(int)));

    double iStart, iElaps;

    // cpu recursive reduction
    iStart = seconds();
    int cpu_sum = cpuRecursiveReduce (tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce\t\telapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);

    // gpu nested reduce kernel
    CHECK(mcMemcpy(d_idata, h_idata, bytes, mcMemcpyHostToDevice));
    iStart = seconds();
    gpuRecursiveReduceNosync<<<grid, block>>>(d_idata, d_odata, block.x);
    CHECK(mcDeviceSynchronize());
    CHECK(mcGetLastError());
    iElaps = seconds() - iStart;
    CHECK(mcMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                        mcMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu nested\t\telapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x, block.x);

    // gpu nested reduce kernel2
    CHECK(mcMemcpy(d_idata, h_idata, bytes, mcMemcpyHostToDevice));
    iStart = seconds();
    gpuRecursiveReduce2<<<grid, block.x / 2>>>(d_idata, d_odata, block.x/2, block.x);
    CHECK(mcDeviceSynchronize());
    CHECK(mcGetLastError());
    iElaps = seconds() - iStart;
    CHECK(mcMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                        mcMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu nested kernel 2\t\telapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n",
            iElaps, gpu_sum, grid.x, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(mcFree(d_idata));
    CHECK(mcFree(d_odata));

    // reset device
    CHECK(mcDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}