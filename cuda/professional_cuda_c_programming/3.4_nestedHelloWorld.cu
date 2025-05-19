#include "./common.h"
#include <__clang_maca_builtin_vars.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void nestedHelloWorld(int const iSzie, int iDepth) 
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d,"
        "block %d\n", iDepth, tid, blockIdx.x); 

    if (iSzie == 1) return;

    int nthreads = iSzie >> 1;

    if (tid == 0 && nthreads > 0) {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main() {
    nestedHelloWorld<<<1, 8>>>(8, 0);
    CHECK(mcGetLastError());
    CHECK(mcGetLastError());
}