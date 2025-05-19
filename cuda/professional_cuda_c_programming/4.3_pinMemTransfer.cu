#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

int main() 
{
    int dev = 0;
    mcSetDevice(dev);

    unsigned int isize = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);

    mcDeviceProp_t deviceProp;
    mcGetDeviceProperties(&deviceProp, dev);
    printf("device %d: %s memory size %d nbytes %5.2fMB\n", 
        dev, deviceProp.name, isize, nbytes/(1024.0f*1024.0f));

    // float *h_a = (float *)malloc(nbytes);
    float *h_a;
    mcMallocHost((float **)&h_a, nbytes);

    float *d_a;
    mcMalloc((float **)&d_a, nbytes);

    for (unsigned int i=0; i<isize; i++) h_a[i] = 0.5f;

    mcMemcpy(d_a, h_a, nbytes, mcMemcpyHostToDevice);

    mcMemcpy(h_a, d_a, nbytes, mcMemcpyDeviceToHost);

    mcFree(d_a);
    mcFreeHost(h_a);

    mcDeviceReset();
    return EXIT_SUCCESS;
}