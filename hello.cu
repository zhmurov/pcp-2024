#include <stdio.h>

__global__ void gpu_kernel()
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    printf("%d\n", i);
}

int main()
{
    gpu_kernel<<<4, 16>>>();
    cudaDeviceSynchronize();
}