#include <iostream>


__global__ void vec_add_kernel(float* d_x, float* d_y, float* d_z, int N)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < N)
    {
        d_z[i] = d_x[i] + d_y[i];
    }
}

int main()
{
    int N = 256;
    float* h_x = (float*)calloc(N, sizeof(float));
    float* h_y = (float*)calloc(N, sizeof(float));
    float* h_z = (float*)calloc(N, sizeof(float));

    float* d_x;
    float* d_y;
    float* d_z;

    cudaMalloc((void**)&d_x, N*sizeof(float));
    cudaMalloc((void**)&d_y, N*sizeof(float));
    cudaMalloc((void**)&d_z, N*sizeof(float));

    for (int i = 0; i < N; i++)
    {
        h_x[i] = i*0.1;
        h_y[i] = i*0.01;
    }

    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);

///// HPC ZONE
    vec_add_kernel<<<N/16 + 1, 16>>>(d_x, d_y, d_z, N);
///// HPC ZONE

    cudaMemcpy(h_z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i ++)
    {
        std::cout << h_z[i] << std::endl;
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    free(h_x);
    free(h_y);
    free(h_z);

    //gpu_kernel<<<4, 16>>>();
    //cudaDeviceSynchronize();
}