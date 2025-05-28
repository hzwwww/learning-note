#include "./common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates the impact of misaligned reads on performance by
 * forcing misaligned reads to occur on a float*.
 */

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                    gpuRef[i]);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void initialData(float *ip,  int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 100.0f;
    }

    return;
}


void sumArraysOnHost(float *A, float *B, float *C, const int n, int offset)
{
    for (int idx = offset, k = 0; idx < n; idx++, k++)
    {
        C[k] = A[idx] + B[idx];
    }
}

__global__ void warmup(float *A, float *B, float *C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

__global__ void readOffset(float *A, float *B, float *C, const int n,
                           int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[i] = A[k] + B[k];
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    mcDeviceProp_t deviceProp;
    CHECK(mcGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(mcSetDevice(dev));

    // set up array size
    int nElem = 1 << 20; // total number of elements to reduce
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // set up offset for summary
    int blocksize = 512;
    int offset = 0;

    if (argc > 1) offset    = atoi(argv[1]);

    if (argc > 2) blocksize = atoi(argv[2]);

    // execution configuration
    dim3 block (blocksize, 1);
    dim3 grid  ((nElem + block.x - 1) / block.x, 1);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    //  initialize host array
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    //  summary at host side
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(mcMalloc((float**)&d_A, nBytes));
    CHECK(mcMalloc((float**)&d_B, nBytes));
    CHECK(mcMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(mcMemcpy(d_A, h_A, nBytes, mcMemcpyHostToDevice));
    CHECK(mcMemcpy(d_B, h_A, nBytes, mcMemcpyHostToDevice));

    //  kernel 1:
    double iStart = seconds();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(mcDeviceSynchronize());
    double iElaps = seconds() - iStart;
    printf("warmup     <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
           block.x, offset, iElaps);
    CHECK(mcGetLastError());

    iStart = seconds();
    readOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(mcDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("readOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
           block.x, offset, iElaps);
    CHECK(mcGetLastError());

    // copy kernel result back to host side and check device results
    CHECK(mcMemcpy(gpuRef, d_C, nBytes, mcMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nElem - offset);

    // free host and device memory
    CHECK(mcFree(d_A));
    CHECK(mcFree(d_B));
    CHECK(mcFree(d_C));
    free(h_A);
    free(h_B);

    // reset device
    CHECK(mcDeviceReset());
    return EXIT_SUCCESS;
}