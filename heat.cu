/*
 * Based on CSC materials from:
 * 
 * https://github.com/csc-training/openacc/tree/master/exercises/heat
 *
 */
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pngwriter.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

/* Convert 2D index layout to unrolled 1D layout
 *
 * \param[in] i      Row index
 * \param[in] j      Column index
 * \param[in] width  The width of the area
 * 
 * \returns An index in the unrolled 1D array.
 */
__host__ __device__ int getIndex(const int i, const int j, const int width)
{
    return i*width + j;
}

__global__ void heat_kernel(int nx, int ny, float* d_Un, float* d_Unp1, float aTimesDt, float dx2, float dy2)
{
    // Going through the entire area
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i > 0 && i < nx-1)
    {
        int j = threadIdx.y + blockIdx.y*blockDim.y;
        if (j > 0 && j < ny-1)
        {
            const int index = getIndex(i, j, ny);
            float uij = d_Un[index];
            float uim1j = d_Un[getIndex(i-1, j, ny)];
            float uijm1 = d_Un[getIndex(i, j-1, ny)];
            float uip1j = d_Un[getIndex(i+1, j, ny)];
            float uijp1 = d_Un[getIndex(i, j+1, ny)];

            // Explicit scheme
            d_Unp1[index] = uij + aTimesDt * ( (uim1j - 2.0*uij + uip1j)/dx2 + (uijm1 - 2.0*uij + uijp1)/dy2 );
        }
    }
}


int main()
{
    const int nx = 200;   // Width of the area
    const int ny = 200;   // Height of the area

    const float a = 0.5;     // Diffusion constant

    const float dx = 0.01;   // Horizontal grid spacing 
    const float dy = 0.01;   // Vertical grid spacing

    const float dx2 = dx*dx;
    const float dy2 = dy*dy;

    const float dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2)); // Largest stable time step
    const int numSteps = 5000;                             // Number of time steps
    const int outputEvery = 1000;                          // How frequently to write output image

    int numElements = nx*ny;

    // Allocate two sets of data for current and next timesteps
    float* h_Un   = (float*)calloc(numElements, sizeof(float));
    float* h_Unp1 = (float*)calloc(numElements, sizeof(float));

    float* d_Un;
    float* d_Unp1;

    cudaMalloc((void**)&d_Un, numElements*sizeof(float));
    cudaMalloc((void**)&d_Unp1, numElements*sizeof(float));

    // Initializing the data with a pattern of disk of radius of 1/6 of the width
    float radius2 = (nx/6.0) * (nx/6.0);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            int index = getIndex(i, j, ny);
            // Distance of point i, j from the origin
            float ds2 = (i - nx/2) * (i - nx/2) + (j - ny/2)*(j - ny/2);
            if (ds2 < radius2)
            {
                h_Un[index] = 65.0;
            }
            else
            {
                h_Un[index] = 5.0;
            }
        }
    }

    // Fill in the data on the next step to ensure that the boundaries are identical.
    memcpy(h_Unp1, h_Un, numElements*sizeof(float));

    cudaMemcpy(d_Un, h_Un, numElements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Unp1, d_Un, numElements*sizeof(float), cudaMemcpyDeviceToDevice);

    // Timing
    clock_t start = clock();

    dim3 numBlocks(nx/BLOCK_SIZE_X + 1, ny/BLOCK_SIZE_Y + 1);
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Main loop
    for (int n = 0; n <= numSteps; n++)
    {
        heat_kernel<<<numBlocks, threadsPerBlock>>>(nx, ny, d_Un, d_Unp1, a*dt, dx2, dy2);
        // Write the output if needed
        if (n % outputEvery == 0)
        {
            cudaMemcpy(h_Un, d_Un, numElements*sizeof(float), cudaMemcpyDeviceToHost);
            char filename[64];
            sprintf(filename, "heat_%04d.png", n);
            save_png(h_Un, nx, ny, filename, 'c');
        }
        // Swapping the pointers for the next timestep
        std::swap(d_Un, d_Unp1);
    }

    // Timing
    clock_t finish = clock();
    printf("It took %f seconds\n", (double)(finish - start) / CLOCKS_PER_SEC);

    // Release the memory
    cudaFree(d_Un);
    cudaFree(d_Unp1);
    free(h_Un);
    free(h_Unp1);
    
    return 0;
}