// 10,000 threads incrementing 10 array elements
// We need to avoid the conflicts
#include <stdio.h>
//#include "gputimer.h"

#define NUM_THREADS 1000
#define ARRAY_SIZE 10

#define BLOCK_WIDTH 100

void print_array(int* array, int size)
{
    for(int i(0); i < size ; ++i)
    {
        printf("%d \t", array[i]);
    }
}

__global__ void increment_naive(int* g)
{
    // thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // map it to the array_size
    i = i % ARRAY_SIZE;
    //g[i] = g[i] + 1;  -> do not do this
    atomicAdd(&g[i], 1);
}

int main()
{
    //Gputimer timer;
    printf("%d total threads in %d blocks writing into %d array element \t",
            NUM_THREADS, NUM_THREADS / BLOCK_WIDTH, ARRAY_SIZE);

    // declare and allocate host memory
    int h_array[ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    // declare, allocate, and zero-oout GPU memory
    int* d_array;
    cudaMalloc((void**) &d_array, ARRAY_BYTES);
    cudaMemset((void*) d_array, 0, ARRAY_BYTES);

    // launch kernel
    //dim3 gridSize = (NUM_THREADS/BLOCK_WIDTH, 1, 1);
    //dim3 blockSize = (BLOCK_WIDTH, 1, 1);

    //timer.Start();
    increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    //timer.Stop();

    // copy back from the GPU memory to CPU memory
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    print_array(h_array, ARRAY_SIZE);

    // check time
    //printf("Time elapsed = %g ms \n", timer.Elapsed());

    // free cuda memory
    cudaFree(d_array);

    return 0;

}